"""
금융감독원 제재 문서 파싱 모듈
- PDF, HWP, HWPX 파일 파싱
- 제재 내용을 구조화된 JSON으로 변환
"""

import os
import json
import re
import pathlib
import shutil
import tempfile
import zipfile
import subprocess
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
#import hwp5
#from hwp5 import hwp5txt
import olefile
import io


def clean_text(text: str) -> str:
    """텍스트 정리"""
    # 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 중 필요한 것만 유지
    text = re.sub(r'[^\w가-힣 .,?!:;()\-\[\]%·※○●□■◇◎△▲▽▼\n]', '', text)
    return text.strip()


def extract_text_from_hwp(hwp_file_path: str) -> str:
    """pyhwp Python API를 사용하여 HWP 파일에서 텍스트 추출"""
    try:
        from hwp5.hwp5txt import Hwp5File, TextTransform
        from hwp5.errors import InvalidHwp5FileError
        from contextlib import closing

        text_transform = TextTransform()
        transform = text_transform.transform_hwp5_to_text

        output = io.BytesIO()
        with closing(Hwp5File(hwp_file_path)) as hwp5file:
            transform(hwp5file, output)

        text_content = output.getvalue().decode('utf-8', errors='ignore')

        if text_content and text_content.strip():
            return text_content.strip()
        else:
            return extract_text_from_hwp_alternative(hwp_file_path)

    except Exception as e:
        return extract_text_from_hwp_alternative(hwp_file_path)


def extract_text_from_hwp_alternative(hwp_file_path: str) -> str:
    """olefile을 사용한 HWP 파일 텍스트 추출 대체 방법 (zlib 압축 해제 포함)"""
    import zlib
    try:
        ole = olefile.OleFileIO(hwp_file_path)

        # BodyText 하위 Section 스트림 탐색
        text_streams = []
        for entry in ole.listdir():
            if len(entry) >= 2 and entry[0] == 'BodyText' and entry[1].startswith('Section'):
                text_streams.append(entry)

        text_streams.sort(key=lambda x: x[1])  # Section0, Section1, ... 순서 정렬

        extracted_text = []
        for stream_path in text_streams:
            try:
                stream_data = ole.openstream(stream_path)
                raw = stream_data.read()
                stream_data.close()

                # HWP BodyText 스트림은 zlib deflate 압축 (앞 4바이트 = 원본 크기)
                try:
                    decompressed = zlib.decompress(raw[4:], -15)
                except zlib.error:
                    decompressed = raw  # 압축 안 된 경우 fallback

                # HWP 레코드에서 HWPTAG_PARA_TEXT(0x42=66) 파싱
                paragraphs = []
                i = 0
                while i + 4 <= len(decompressed):
                    header = int.from_bytes(decompressed[i:i+4], 'little')
                    tag_id = header & 0x3FF
                    record_size = (header >> 20) & 0xFFF
                    if record_size == 0xFFF and i + 8 <= len(decompressed):
                        record_size = int.from_bytes(decompressed[i+4:i+8], 'little')
                        i += 4
                    i += 4
                    if i + record_size > len(decompressed):
                        break
                    if tag_id == 66:  # HWPTAG_PARA_TEXT
                        para_bytes = decompressed[i:i+record_size]
                        para_text = para_bytes.decode('utf-16le', errors='ignore')
                        # 특수 제어 문자(0x0D=단락 끝 등) 제거
                        para_text = re.sub(r'[\x00-\x1f]', '', para_text)
                        if para_text.strip():
                            paragraphs.append(para_text.strip())
                    i += record_size

                if paragraphs:
                    extracted_text.append('\n'.join(paragraphs))

            except Exception:
                continue

        ole.close()

        full_text = '\n'.join(extracted_text)
        if full_text.strip():
            return full_text
        else:
            return "HWP 파일 텍스트 추출 실패"

    except Exception as e:
        print(f"olefile 파싱 실패: {e}")
        return "HWP 파일 파싱 실패"


def hwp2html(hwp_file_path: str, html_file_dir: str) -> bool:
    """HWP 파일을 HTML로 변환"""
    try:
        # hwp5html 명령어 시도
        result = os.system(f'hwp5html --output "{html_file_dir}" "{hwp_file_path}"')
        return result == 0
    except:
        return False


class FSSDocumentParser:
    """금융감독원 제재 문서 파싱 클래스"""
    
    def __init__(self):
        self.sanction_patterns = {
            'institution': [
                r'금융회사명\s*[:：]\s*([^\n]+)',
                r'제재대상기관\s*[:：]\s*([^\n]+)',
                r'회사명\s*[:：]\s*([^\n]+)'
            ],
            'date': [
                r'제재조치일\s*[:：]\s*(\d{4}[\.\s]*\d{1,2}[\.\s]*\d{1,2})',
                r'제재일자\s*[:：]\s*(\d{4}[\.\s]*\d{1,2}[\.\s]*\d{1,2})'
            ],
            'sanction_type': [
                r'제재내용\s*[:：]?\s*([^\n]+)',
                r'제재조치내용\s*[:：]?\s*([^\n]+)'
            ],
            'fine': [
                r'과태료\s*[:：]?\s*([0-9,]+)\s*백만원',
                r'과태료\s*[:：]?\s*([0-9,]+)\s*만원',
                r'과태료\s*[:：]?\s*([0-9,]+)\s*원'
            ],
            'executive_sanction': [
                r'임원\s*[:：]?\s*([^\n]+)',
                r'임직원\s*[:：]?\s*([^\n]+)'
            ]
        }
    
    def extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일에서 제재 정보 추출"""
        try:
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n"
            
            doc.close()
            
            return self.parse_sanction_content(full_text, file_path)
            
        except Exception as e:
            print(f"[오류] PDF 파싱 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def extract_from_hwp(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일에서 제재 정보 추출"""
        try:
            print(f"[처리] HWP 파일 파싱 시작: {os.path.basename(file_path)}")
            
            # pyhwp를 사용한 직접 텍스트 추출
            full_text = extract_text_from_hwp(file_path)
            
            if full_text and full_text.strip() and "실패" not in full_text:
                print(f"[성공] HWP 텍스트 추출 성공: {len(full_text)} 문자")
                return self.parse_sanction_content(full_text, file_path)
            else:
                print(f"[경고] HWP 직접 파싱 실패, HTML 변환 시도")
                
                # HTML 변환 방법 시도
                hwp_file_dir = os.path.dirname(file_path)
                hwp_file_name = os.path.basename(file_path)
                html_file_dir = os.path.join(hwp_file_dir, hwp_file_name.split('.')[0])
                html_file_path = os.path.join(html_file_dir, 'index.xhtml')
                
                conversion_success = hwp2html(file_path, html_file_dir)
                
                if conversion_success and os.path.exists(html_file_path):
                    # HTML 파싱
                    with open(html_file_path, 'rt', encoding='utf-8') as f:
                        page = BeautifulSoup(f.read(), 'html.parser')
                    
                    # 임시 디렉토리 삭제
                    if os.path.exists(html_file_dir):
                        shutil.rmtree(html_file_dir)
                    
                    full_text = page.get_text(separator='\n')
                    print(f"[성공] HWP HTML 변환 성공: {len(full_text)} 문자")
                    
                    return self.parse_sanction_content(full_text, file_path)
                else:
                    print(f"[경고] HTML 변환도 실패, 기본 정보만 추출")
                    return self.extract_hwp_alternative(file_path)
            
        except Exception as e:
            print(f"[오류] HWP 파싱 에러 {file_path}: {e}")
            return self.extract_hwp_alternative(file_path)
    
    def extract_hwp_alternative(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일 대체 처리 방법"""
        try:
            # 파일 정보만으로 기본 구조 생성
            file_name = os.path.basename(file_path)
            doc_id = os.path.splitext(file_name)[0]
            
            # 파일명에서 정보 추출
            institution = "미확인"
            date = "미확인"
            
            # 파일명에서 날짜 추출 시도
            date_match = re.search(r'(\d{8})', file_name)
            if date_match:
                date_str = date_match.group(1)
                date = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
            
            # 파일명에서 기관명 추출 시도
            inst_match = re.search(r'SANCTION_\d+_([^_]+)_', file_name)
            if inst_match:
                institution = inst_match.group(1).replace('_', ' ')
            
            # 기본 결과 구성
            result = {
                "doc_id": doc_id,
                "source_file": file_name,
                "institution": institution,
                "date": date,
                "content": {
                    "full_text": "HWP 파일 변환 도구가 필요합니다. 수동으로 내용을 확인해주세요."
                },
                "metadata": {
                    "doc_type": "제재내용공개",
                    "char_count": 0,
                    "estimated_tokens": 0,
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "regulations": [],
                    "conversion_status": "변환 실패"
                },
                "llm_metadata": {
                    "keywords": [],
                    "regulations": "",
                    "fines": "",
                    "executive_sanction": ""
                },
                "quality_score": 1,
                "status": "HWP 변환 필요"
            }
            
            print(f"[파일] HWP 파일 기본 정보만 추출: {institution} ({date})")
            return result
            
        except Exception as e:
            print(f"[오류] HWP 대체 처리 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def extract_from_hwpx(self, file_path: str) -> Dict[str, Any]:
        """HWPX 파일에서 제재 정보 추출"""
        try:
            full_text = ""
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # HWPX 파일을 ZIP으로 압축해제
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # section XML 파일들 찾기
                contents_dir = os.path.join(temp_dir, 'Contents')
                section_files = [f for f in os.listdir(contents_dir) if f.startswith('section') and f.endswith('.xml')]
                
                for section_file in sorted(section_files):
                    section_path = os.path.join(contents_dir, section_file)
                    
                    with open(section_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    
                    # BeautifulSoup으로 XML 파싱
                    soup = BeautifulSoup(xml_content, 'lxml-xml')
                    
                    # 텍스트 추출
                    text_elements = soup.find_all(['hp:t'])
                    for elem in text_elements:
                        text = elem.get_text(strip=True)
                        if text:
                            full_text += text + "\n"
            
            return self.parse_sanction_content(full_text, file_path)
            
        except Exception as e:
            print(f"[오류] HWPX 파싱 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def parse_sanction_content(self, text: str, file_path: str) -> Dict[str, Any]:
        """제재 내용 파싱"""
        # 기본 정보 추출
        institution = self.extract_pattern(text, self.sanction_patterns['institution'])
        date = self.extract_pattern(text, self.sanction_patterns['date'])
        sanction_type = self.extract_pattern(text, self.sanction_patterns['sanction_type'])
        
        # 과태료 정보 추출
        fine_info = self.extract_fine(text)
        
        # 임원 제재 정보 추출
        executive_sanction = self.extract_pattern(text, self.sanction_patterns['executive_sanction'])
        
        # 제재 사실 추출
        facts = self.extract_sanction_facts(text)
        
        # 관련 규정 추출
        regulations = self.extract_regulations(text)
        
        # 파일명에서 정보 추출
        file_name = os.path.basename(file_path)
        doc_id = os.path.splitext(file_name)[0]
        
        # 파일명에서 날짜 추출 시도
        if not date:
            date_match = re.search(r'(\d{8})', file_name)
            if date_match:
                date_str = date_match.group(1)
                date = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
        
        # 파일명에서 기관명 추출 시도
        if not institution:
            inst_match = re.search(r'SANCTION_\d+_([^_]+)_', file_name)
            if inst_match:
                institution = inst_match.group(1).replace('_', ' ')
        
        # 결과 구성
        result = {
            "doc_id": doc_id,
            "source_file": file_name,
            "institution": institution or "미확인",
            "date": date or "미확인",
            "content": {
                "full_text": clean_text(text[:1000]) + "..." if len(text) > 1000 else clean_text(text)
            },
            "metadata": {
                "doc_type": "제재내용공개",
                "char_count": len(text),
                "estimated_tokens": len(text.split()),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "regulations": regulations
            },
            "llm_metadata": {
                "keywords": [],
                "regulations": "",
                "fines": fine_info['text'] if fine_info['amount'] > 0 else "",
                "executive_sanction": executive_sanction or ""
            },
            "quality_score": self.calculate_quality_score(institution, date, sanction_type, facts),
            "status": "처리완료"
        }
        
        return result
    
    def extract_pattern(self, text: str, patterns: List[str]) -> Optional[str]:
        """패턴 매칭으로 정보 추출"""
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return clean_text(match.group(1))
        return None
    
    def extract_fine(self, text: str) -> Dict[str, Any]:
        """과태료 정보 추출"""
        fine_info = {
            "amount": 0,
            "unit": "원",
            "text": ""
        }
        
        # 과태료 패턴 매칭
        patterns = [
            (r'과태료\s*[:：]?\s*([0-9,]+)\s*백만원', 1000000),
            (r'과태료\s*[:：]?\s*([0-9,]+)\s*만원', 10000),
            (r'과태료\s*[:：]?\s*([0-9,]+)\s*원', 1)
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                fine_info['amount'] = int(amount_str) * multiplier
                fine_info['text'] = match.group(0)
                if multiplier == 1000000:
                    fine_info['unit'] = "백만원"
                elif multiplier == 10000:
                    fine_info['unit'] = "만원"
                break
        
        return fine_info
    
    def extract_sanction_facts(self, text: str) -> List[Dict[str, str]]:
        """제재 사실 추출"""
        facts = []
        
        # 제재 사실 섹션 찾기
        fact_patterns = [
            r'제재대상사실\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)',
            r'위반사항\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)',
            r'위반내용\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)'
        ]
        
        # 항목별 패턴 (가. 나. 다. 또는 (1) (2) (3) 형식)
        item_patterns = [
            r'([가-하]\.\s*[^\n]+(?:\n(?![가-하]\.|[0-9]+\.|\<관련규정\>)[^\n]+)*)',
            r'(\([0-9]+\)\s*[^\n]+(?:\n(?!\([0-9]+\)|[가-하]\.|\<관련규정\>)[^\n]+)*)'
        ]
        
        for pattern in fact_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                fact_section = match.group(1)
                
                # 항목별로 분리
                for item_pattern in item_patterns:
                    items = re.findall(item_pattern, fact_section, re.MULTILINE | re.DOTALL)
                    for item in items:
                        # 제목과 내용 분리
                        lines = item.strip().split('\n', 1)
                        if lines:
                            title = clean_text(lines[0])
                            content = clean_text(lines[1]) if len(lines) > 1 else ""
                            
                            facts.append({
                                "title": title,
                                "content": content
                            })
        
        # 패턴 매칭 실패시 간단한 분할
        if not facts:
            # 전자금융거래 관련 키워드로 섹션 찾기
            keywords = [
                '전자금융',
                '정보처리위탁',
                '정보처리 업무 위탁에 관한 규정',
                '신용정보법',
                '신용정보의 이용 및 보호에 관한 법률',
                '신용정보업감독규정',
                '신용정보업감독업무시행세칙'
            ]
            for keyword in keywords:
                if keyword in text:
                    # 키워드 주변 텍스트 추출
                    idx = text.find(keyword)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + 500)
                    snippet = text[start:end]
                    
                    facts.append({
                        "title": f"{keyword} 관련 위반",
                        "content": clean_text(snippet)
                    })
                    
                    if len(facts) >= 3:  # 최대 3개까지만
                        break
        
        return facts
    
    def extract_regulations(self, text: str) -> List[str]:
        """관련 규정 추출"""
        regulations = []
        
        # 관련규정 섹션 찾기
        reg_section_match = re.search(r'<관련규정>(.*?)(?=<|$)', text, re.DOTALL)
        if reg_section_match:
            reg_section = reg_section_match.group(1)
        else:
            reg_section = text
        
        # 법령 패턴
        law_patterns = [
            r'「([^」]+)」\s*제(\d+)조',
            r'([가-힣]+법)\s*제(\d+)조',
            r'([가-힣]+규정)\s*제(\d+)조',
            r'([가-힣]+규칙)\s*제(\d+)조'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, reg_section)
            for match in matches:
                law_name = match[0]
                article = f"제{match[1]}조"
                regulation = f"{law_name} {article}"
                if regulation not in regulations:
                    regulations.append(regulation)
        
        return regulations[:10]  # 최대 10개까지
    
    
    def calculate_quality_score(self, institution: str, date: str, sanction_type: str, facts: List) -> int:
        """품질 점수 계산 (1-5)"""
        score = 1
        
        if institution and institution != "미확인":
            score += 1
        if date and date != "미확인":
            score += 1
        if sanction_type and sanction_type != "미확인":
            score += 1
        if facts and len(facts) > 0:
            score += 1
        
        return min(score, 5)
    
    def create_empty_result(self, file_path: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        file_name = os.path.basename(file_path)
        doc_id = os.path.splitext(file_name)[0]
        
        return {
            "doc_id": doc_id,
            "source_file": file_name,
            "institution": "파싱실패",
            "date": "파싱실패",
            "content": {
                "full_text": ""
            },
            "metadata": {
                "doc_type": "제재내용공개",
                "char_count": 0,
                "estimated_tokens": 0,
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "regulations": []
            },
            "llm_metadata": {
                "keywords": [],
                "regulations": "",
                "fines": "",
                "executive_sanction": ""
            },
            "quality_score": 0,
            "status": "파싱실패"
        }
    
    def process_directory(self, input_dir: str, output_json: str) -> Dict[str, Any]:
        """디렉토리 내 모든 제재 문서 처리"""
        results = []
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "file_types": {"pdf": 0, "hwp": 0, "hwpx": 0}
        }
        
        if not os.path.exists(input_dir):
            print(f"[오류] 디렉토리가 존재하지 않습니다: {input_dir}")
            return {"data": results, "stats": stats}
        
        # 파일 목록 가져오기
        files = []
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(('.pdf', '.hwp', '.hwpx')):
                files.append(os.path.join(input_dir, file_name))
        
        stats["total_files"] = len(files)
        print(f"\n[폴더] 총 {len(files)}개 파일 발견")
        
        # 각 파일 처리
        for file_path in sorted(files):
            file_name = os.path.basename(file_path)
            ext = pathlib.Path(file_path).suffix.lower()
            
            print(f"\n[검색] 처리 중: {file_name}")
            
            try:
                if ext == '.pdf':
                    result = self.extract_from_pdf(file_path)
                    stats["file_types"]["pdf"] += 1
                elif ext == '.hwp':
                    result = self.extract_from_hwp(file_path)
                    stats["file_types"]["hwp"] += 1
                elif ext == '.hwpx':
                    result = self.extract_from_hwpx(file_path)
                    stats["file_types"]["hwpx"] += 1
                else:
                    continue
                
                if result["status"] == "파싱실패":
                    stats["failed_files"] += 1
                else:
                    stats["processed_files"] += 1
                
                results.append(result)
                print(f"[성공] 완료: {result['institution']} ({result['date']})")
                
            except Exception as e:
                print(f"[오류] 에러: {file_name} - {e}")
                stats["failed_files"] += 1
                results.append(self.create_empty_result(file_path))
        
        # JSON 저장
        output_data = {
            "metadata": {
                "source": "금융감독원 제재내용공개",
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(results)
            },
            "statistics": stats,
            "documents": results
        }
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[완료] 완료!")
        print(f"[통계] 통계:")
        print(f"  - 전체 파일: {stats['total_files']}개")
        print(f"  - 성공: {stats['processed_files']}개")
        print(f"  - 실패: {stats['failed_files']}개")
        print(f"  - PDF: {stats['file_types']['pdf']}개")
        print(f"  - HWP: {stats['file_types']['hwp']}개")
        print(f"  - HWPX: {stats['file_types']['hwpx']}개")
        print(f"\n[저장] 결과 저장: {output_json}")
        
        return output_data


# 사용 예시
if __name__ == "__main__":
    parser = FSSDocumentParser()
    
    # 입력 디렉토리와 출력 파일 설정
    input_directory = "../data/FSS_SANCTION/data/FSS_SANCTION"
    output_file = "./data/fss_sanctions_parsed.json"
    
    # 디렉토리 처리
    result = parser.process_directory(input_directory, output_file)