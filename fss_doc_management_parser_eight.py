"""
금융감독원 경영유의 문서 파싱 모듈
- PDF, HWP, HWPX 파일 파싱
- 경영유의 내용을 구조화된 JSON으로 변환
- 지정된 키워드가 포함된 문서만 코퍼스에 포함
"""

import os
import json
import re
import pathlib
import shutil
import tempfile
import zipfile
import subprocess
import fitz 
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any


def clean_text(text: str) -> str:
    """텍스트 정리"""
    # 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 중 필요한 것만 유지
    text = re.sub(r'[^\w가-힣 .,?!:;()\-\[\]%·※○●□■◇◎△▲▽▼\n]', '', text)
    return text.strip()


def extract_text_from_hwp_mgmt(hwp_file_path: str) -> str:
    """hwp5txt 명령줄 도구를 사용하여 HWP 파일에서 텍스트 추출"""
    try:
        # 임시 텍스트 파일 경로
        temp_txt_path = hwp_file_path + '.txt'
        
        # hwp5txt 명령 실행
        command = f'hwp5txt "{hwp_file_path}" --output "{temp_txt_path}"'
        result = os.system(command)
        
        if result == 0 and os.path.exists(temp_txt_path):
            # 텍스트 파일 읽기
            with open(temp_txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # 임시 파일 삭제
            os.remove(temp_txt_path)
            
            if text_content and text_content.strip():
                print(f"✅ hwp5txt로 텍스트 추출 성공: {len(text_content)} 문자")
                return text_content.strip()
            else:
                print(f"⚠️ hwp5txt 결과 비어있음, 대체 방법 시도")
                return extract_text_from_hwp_alternative_mgmt(hwp_file_path)
        else:
            print(f"⚠️ hwp5txt 명령 실패 (코드: {result}), 대체 방법 시도")
            return extract_text_from_hwp_alternative_mgmt(hwp_file_path)
            
    except Exception as e:
        print(f"⚠️ hwp5txt 파싱 실패: {e}")
        # 대체 방법 시도
        return extract_text_from_hwp_alternative_mgmt(hwp_file_path)


def extract_text_from_hwp_alternative_mgmt(hwp_file_path: str) -> str:
    """olefile을 사용한 HWP 파일 텍스트 추출 대체 방법"""
    try:
        import olefile
        
        # OLE 파일로 HWP 파일 열기
        ole = olefile.OleFileIO(hwp_file_path)
        
        # 텍스트 스트림 찾기
        text_streams = []
        for stream_name in ole.listdir():
            if isinstance(stream_name, list) and len(stream_name) > 0:
                if 'BodyText' in stream_name[0] or 'Section' in stream_name[0]:
                    text_streams.append(stream_name)
        
        # 텍스트 추출
        extracted_text = []
        for stream in text_streams:
            try:
                with ole.open(stream) as stream_data:
                    # 바이너리 데이터 읽기
                    data = stream_data.read()
                    
                    # 유니코드 텍스트 추출 시도
                    try:
                        # UTF-16 시도
                        text = data.decode('utf-16le', errors='ignore')
                        # 제어 문자 제거
                        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                        if text.strip():
                            extracted_text.append(text.strip())
                    except:
                        # CP949 시도
                        try:
                            text = data.decode('cp949', errors='ignore')
                            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                            if text.strip():
                                extracted_text.append(text.strip())
                        except:
                            pass
            except:
                continue
        
        ole.close()
        
        full_text = '\n'.join(extracted_text)
        if full_text.strip():
            return full_text
        else:
            return "HWP 파일 텍스트 추출 실패"
            
    except Exception as e:
        print(f"⚠️ olefile 파싱도 실패: {e}")
        return "HWP 파일 파싱 실패"


def hwp2html(hwp_file_path: str, html_file_dir: str) -> bool:
    """HWP 파일을 HTML로 변환"""
    try:
        # hwp5html 명령어 시도
        result = os.system(f'hwp5html --output "{html_file_dir}" "{hwp_file_path}"')
        return result == 0
    except:
        return False


def is_electronic_finance_related(content: str, filename: str = "", institution: str = "") -> bool:
    """지정된 키워드로 전자금융 관련 여부 판단"""
    keywords = [
        '전자금융',
        '정보처리위탁',
        '정보처리 업무 위탁에 관한 규정',
        '신용정보법',
        '신용정보의 이용 및 보호에 관한 법률',
        '신용정보업감독규정',
        '신용정보업감독업무시행세칙'
    ]
    
    # 문서 내용에서 키워드 확인
    for keyword in keywords:
        if keyword in content:
            return True
    
    # 파일명에서 키워드 확인
    for keyword in keywords:
        if keyword in filename:
            return True
    
    # 기관명에서 키워드 확인
    for keyword in keywords:
        if keyword in institution:
            return True
    
    return False


def extract_found_keywords(content: str, filename: str = "", institution: str = "") -> List[str]:
    """발견된 키워드 목록 반환"""
    keywords = [
        '전자금융',
        '정보처리위탁',
        '정보처리 업무 위탁에 관한 규정',
        '신용정보법',
        '신용정보의 이용 및 보호에 관한 법률',
        '신용정보업감독규정',
        '신용정보업감독업무시행세칙'
    ]
    
    found_keywords = []
    
    # 문서 내용, 파일명, 기관명에서 키워드 찾기
    all_text = f"{content} {filename} {institution}"
    
    for keyword in keywords:
        if keyword in all_text:
            found_keywords.append(keyword)
    
    return found_keywords


class FSSManagementParser:
    """금융감독원 경영유의 문서 파싱 클래스"""
    
    def __init__(self):
        self.management_patterns = {
            'institution': [
                r'회사명\s*[:：]\s*([^\n]+)',
                r'기관명\s*[:：]\s*([^\n]+)',
                r'금융회사명\s*[:：]\s*([^\n]+)',
                r'대상기관\s*[:：]\s*([^\n]+)'
            ],
            'date': [
                r'공시일\s*[:：]\s*(\d{4}[\.\-\s]*\d{1,2}[\.\-\s]*\d{1,2})',
                r'공시일자\s*[:：]\s*(\d{4}[\.\-\s]*\d{1,2}[\.\-\s]*\d{1,2})',
                r'작성일\s*[:：]\s*(\d{4}[\.\-\s]*\d{1,2}[\.\-\s]*\d{1,2})',
                r'(\d{4}[\.\-\s]*\d{1,2}[\.\-\s]*\d{1,2})'
            ],
            'management_type': [
                r'경영유의사항\s*[:：]?\s*([^\n]+)',
                r'유의사항\s*[:：]?\s*([^\n]+)',
                r'내용\s*[:：]?\s*([^\n]+)'
            ],
        }
    
    def extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일에서 경영유의 정보 추출"""
        try:
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n"
            
            doc.close()
            
            return self.parse_management_content(full_text, file_path)
            
        except Exception as e:
            print(f"❌ PDF 파싱 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def extract_from_hwp(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일에서 경영유의 정보 추출"""
        try:
            print(f"🔄 HWP 파일 파싱 시작: {os.path.basename(file_path)}")
            
            # hwp5txt를 사용한 직접 텍스트 추출
            full_text = extract_text_from_hwp_mgmt(file_path)
            
            if full_text and full_text.strip() and "파일 파싱 실패" not in full_text:
                print(f"✅ HWP 텍스트 추출 성공: {len(full_text)} 문자")
                return self.parse_management_content(full_text, file_path)
            else:
                print(f"⚠️ HWP 직접 파싱 실패, HTML 변환 시도")
                
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
                    print(f"✅ HWP HTML 변환 성공: {len(full_text)} 문자")
                    
                    return self.parse_management_content(full_text, file_path)
                else:
                    print(f"⚠️ HTML 변환도 실패, 기본 정보만 추출")
                    return self.extract_hwp_alternative(file_path)
            
        except Exception as e:
            print(f"❌ HWP 파싱 에러 {file_path}: {e}")
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
            inst_match = re.search(r'MGMT_\d+_([^_]+)_', file_name)
            if inst_match:
                institution = inst_match.group(1).replace('_', ' ')
            
            # 기본 결과 구성
            result = {
                "doc_id": doc_id,
                "source_file": file_name,
                "institution": institution,
                "date": date,
                "is_relevant": False,
                "found_keywords": [],
                "content": {
                    "full_text": "HWP 파일 변환 도구가 필요합니다. 수동으로 내용을 확인해주세요."
                },
                "metadata": {
                    "doc_type": "경영유의사항",
                    "char_count": 0,
                    "estimated_tokens": 0,
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "regulations": [],
                    "file_extension": pathlib.Path(file_path).suffix.lower(),
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
            
            print(f"📄 HWP 파일 기본 정보만 추출: {institution} ({date})")
            return result
            
        except Exception as e:
            print(f"❌ HWP 대체 처리 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def extract_from_hwpx(self, file_path: str) -> Dict[str, Any]:
        """HWPX 파일에서 경영유의 정보 추출"""
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
            
            return self.parse_management_content(full_text, file_path)
            
        except Exception as e:
            print(f"❌ HWPX 파싱 에러 {file_path}: {e}")
            return self.create_empty_result(file_path)
    
    def parse_management_content(self, text: str, file_path: str) -> Dict[str, Any]:
        """경영유의 내용 파싱"""
        # 기본 정보 추출
        institution = self.extract_pattern(text, self.management_patterns['institution'])
        date = self.extract_pattern(text, self.management_patterns['date'])
        management_type = self.extract_pattern(text, self.management_patterns['management_type'])
        
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
            inst_match = re.search(r'MGMT_\d+_([^_]+)_', file_name)
            if inst_match:
                institution = inst_match.group(1).replace('_', ' ')
        
        # 전자금융 관련 여부 확인
        is_relevant = is_electronic_finance_related(text, file_name, institution or "")
        found_keywords = extract_found_keywords(text, file_name, institution or "")
        
        # 결과 구성
        result = {
            "doc_id": doc_id,
            "source_file": file_name,
            "institution": institution or "미확인",
            "date": date or "미확인",
            "is_relevant": is_relevant,
            "found_keywords": found_keywords,
            "content": {
                "full_text": clean_text(text)
            },
            "metadata": {
                "doc_type": "경영유의사항",
                "char_count": len(text),
                "estimated_tokens": len(text.split()),
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "regulations": regulations,
                "file_extension": pathlib.Path(file_path).suffix.lower()
            },
            "llm_metadata": {
                "keywords": [],
                "regulations": "",
                "fines": "",
                "executive_sanction": ""
            },
            "quality_score": self.calculate_quality_score(institution, date, management_type),
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
    
    def extract_management_details(self, text: str) -> List[Dict[str, str]]:
        """경영유의 상세 내용 추출"""
        details = []
        
        # 경영유의 관련 섹션 찾기
        detail_patterns = [
            r'경영유의사항\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)',
            r'유의사항\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)',
            r'내용\s*[:：]?\s*\n([^\n]+(?:\n(?![0-9]+\.|가\.|나\.|다\.)[^\n]+)*)'
        ]
        
        # 항목별 패턴 (가. 나. 다. 또는 (1) (2) (3) 형식)
        item_patterns = [
            r'([가-하]\.\s*[^\n]+(?:\n(?![가-하]\.|[0-9]+\.)[^\n]+)*)',
            r'(\([0-9]+\)\s*[^\n]+(?:\n(?!\([0-9]+\)|[가-하]\.)[^\n]+)*)',
            r'([0-9]+\.\s*[^\n]+(?:\n(?![0-9]+\.|[가-하]\.)[^\n]+)*)'
        ]
        
        for pattern in detail_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                detail_section = match.group(1)
                
                # 항목별로 분리
                for item_pattern in item_patterns:
                    items = re.findall(item_pattern, detail_section, re.MULTILINE | re.DOTALL)
                    for item in items:
                        # 제목과 내용 분리
                        lines = item.strip().split('\n', 1)
                        if lines:
                            title = clean_text(lines[0])
                            content = clean_text(lines[1]) if len(lines) > 1 else ""
                            
                            details.append({
                                "title": title,
                                "content": content
                            })
        
        # 패턴 매칭 실패시 키워드 기반 추출
        if not details:
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
                    start = max(0, idx - 200)
                    end = min(len(text), idx + 800)
                    snippet = text[start:end]
                    
                    details.append({
                        "title": f"{keyword} 관련 내용",
                        "content": clean_text(snippet)
                    })
                    
                    if len(details) >= 5:  # 최대 5개까지만
                        break
        
        return details
    
    def extract_regulations(self, text: str) -> List[str]:
        """관련 규정 추출"""
        regulations = []
        
        # 법령 패턴
        law_patterns = [
            r'「([^」]+)」\s*제(\d+)조',
            r'([가-힣]+법)\s*제(\d+)조',
            r'([가-힣]+규정)\s*제(\d+)조',
            r'([가-힣]+규칙)\s*제(\d+)조',
            r'([가-힣]+시행세칙)\s*제(\d+)조'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                law_name = match[0]
                article = f"제{match[1]}조"
                regulation = f"{law_name} {article}"
                if regulation not in regulations:
                    regulations.append(regulation)
        
        return regulations[:10]  # 최대 10개까지
    
    def calculate_quality_score(self, institution: str, date: str, management_type: str, details: List = None) -> int:
        """품질 점수 계산 (1-5)"""
        score = 1
        
        if institution and institution != "미확인":
            score += 1
        if date and date != "미확인":
            score += 1
        if management_type and management_type != "미확인":
            score += 1
        if details and len(details) > 0:
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
            "is_relevant": False,
            "found_keywords": [],
            "content": {
                "full_text": ""
            },
            "metadata": {
                "doc_type": "경영유의사항",
                "char_count": 0,
                "estimated_tokens": 0,
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "regulations": [],
                "file_extension": pathlib.Path(file_path).suffix.lower()
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
        """디렉토리 내 모든 경영유의 문서 처리"""
        results = []
        relevant_results = []
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "relevant_files": 0,
            "file_types": {"pdf": 0, "hwp": 0, "hwpx": 0}
        }
        
        if not os.path.exists(input_dir):
            print(f"❌ 디렉토리가 존재하지 않습니다: {input_dir}")
            return {"data": results, "stats": stats}
        
        # 파일 목록 가져오기
        files = []
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(('.pdf', '.hwp', '.hwpx')):
                files.append(os.path.join(input_dir, file_name))
        
        stats["total_files"] = len(files)
        print(f"\n📁 총 {len(files)}개 파일 발견")
        
        # 각 파일 처리
        for file_path in sorted(files):
            file_name = os.path.basename(file_path)
            ext = pathlib.Path(file_path).suffix.lower()
            
            print(f"\n🔍 처리 중: {file_name}")
            
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
                
                # 관련 문서만 별도 저장
                if result.get("is_relevant", False):
                    relevant_results.append(result)
                    stats["relevant_files"] += 1
                    print(f"✅ 관련 문서: {result['institution']} ({result['date']}) - 키워드: {', '.join(result['found_keywords'])}")
                else:
                    print(f"⏭️  일반 문서: {result['institution']} ({result['date']})")
                
            except Exception as e:
                print(f"❌ 에러: {file_name} - {e}")
                stats["failed_files"] += 1
                results.append(self.create_empty_result(file_path))
        
        # 전체 결과 저장
        all_output_data = {
            "metadata": {
                "source": "금융감독원 경영유의사항 공시",
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(results),
                "relevant_documents": len(relevant_results),
                "keywords": [
                    "전자금융",
                    "정보처리위탁",
                    "정보처리 업무 위탁에 관한 규정",
                    "신용정보법",
                    "신용정보의 이용 및 보호에 관한 법률",
                    "신용정보업감독규정",
                    "신용정보업감독업무시행세칙"
                ]
            },
            "statistics": stats,
            "documents": results
        }
        
        # 관련 문서만 코퍼스 저장
        corpus_output_data = {
            "metadata": {
                "source": "금융감독원 경영유의사항 공시",
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(relevant_results),
                "keywords": [
                    "전자금융",
                    "정보처리위탁",
                    "정보처리 업무 위탁에 관한 규정",
                    "신용정보법",
                    "신용정보의 이용 및 보호에 관한 법률",
                    "신용정보업감독규정",
                    "신용정보업감독업무시행세칙"
                ]
            },
            "statistics": {
                "total_processed": stats["processed_files"],
                "relevant_documents": stats["relevant_files"],
                "relevance_rate": round(stats["relevant_files"] / max(stats["processed_files"], 1) * 100, 2)
            },
            "documents": relevant_results
        }
        
        # 전체 결과 저장
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, ensure_ascii=False, indent=2)
        
        # 코퍼스 저장
        corpus_file = output_json.replace('.json', '_corpus.json')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 완료!")
        print(f"📊 통계:")
        print(f"  - 전체 파일: {stats['total_files']}개")
        print(f"  - 성공: {stats['processed_files']}개")
        print(f"  - 실패: {stats['failed_files']}개")
        print(f"  - 관련 문서: {stats['relevant_files']}개")
        print(f"  - 관련도: {round(stats['relevant_files'] / max(stats['processed_files'], 1) * 100, 2)}%")
        print(f"  - PDF: {stats['file_types']['pdf']}개")
        print(f"  - HWP: {stats['file_types']['hwp']}개")
        print(f"  - HWPX: {stats['file_types']['hwpx']}개")
        print(f"\n💾 전체 결과: {output_json}")
        print(f"💾 코퍼스 결과: {corpus_file}")
        
        return all_output_data


# 사용 예시
if __name__ == "__main__":
    parser = FSSManagementParser()
    
    # 입력 디렉토리와 출력 파일 설정
    input_directory = "../../data/FSS_MANAGEMENT"
    output_file = "./data/fss_management_parsed.json"
    
    # 디렉토리 처리
    result = parser.process_directory(input_directory, output_file) 