"""
금융위원회 문서 파싱 모듈
- HWP, HWPX, TXT 파일 파싱
- JSON 변환 및 품질 관리
"""

import os
import json
import re
import pathlib
import shutil
import tempfile
import zipfile
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any


def clean_text(text: str) -> str:
    """텍스트 정리"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w가-힣 .,?!:;\\\-□■○●△▲▽▼※*•–—()\\\[\\\]]', '', text)
    return text.strip()


def hwp2html(hwp_file_path: str, html_file_dir: str) -> None:
    """HWP 파일을 HTML로 변환"""
    os.system(f'hwp5html --output "{html_file_dir}" "{hwp_file_path}"')


class DocumentParser:
    """문서 파싱 클래스"""
    
    def __init__(self):
        pass
    
    def extract_from_txt(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        """TXT 파일에서 질의요지, 회답, 이유 추출"""
        # 특수 케이스 제외
        if 'NAL_1761' in file_path:
            print(f"❌ 제외: {file_path} (신청인 철회)")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 패턴 정의 (띄어쓰기/개행 허용)
        patterns = {
            'question': [
                r'(요청\s*대상\s*행위)[\s:：]*([\\s\\S]*?)(?=(\n\s*(질의\s*요지|회답|판단|이유|판단\s*이유|판단이유)|$))',
                r'(질의\s*요지)[\s:：]*([\\s\\S]*?)(?=(\n\s*(요청\s*대상\s*행위|회답|판단|이유|판단\s*이유|판단이유)|$))'
            ],
            'answer': [
                r'(회답|판단)[\s:：]*([\\s\\S]*?)(?=(\n\s*(이유|판단\s*이유|판단이유)|$))'
            ],
            'reason': [
                r'(이유|판단\s*이유|판단이유)[\s:：]*([\\s\\S]*?)(?=$)'
            ]
        }

        def extract(patterns):
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return m.group(2).strip()
            return ""

        question = extract(patterns['question'])
        answer = extract(patterns['answer'])
        reason = extract(patterns['reason'])

        # "요청을 반려합니다" 필터링
        if '요청을 반려합니다' in answer or '요청을 반려합니다' in reason:
            print(f"❌ 제외: {file_path} (요청을 반려합니다 포함)")
            return None

        return clean_text(question), clean_text(answer), clean_text(reason)

    def extract_from_hwp(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        """HWP 파일에서 질의요지, 회답, 이유 추출"""
        # 특수 케이스 처리
        if 'LAW_3880' in file_path:
            hwp_file_dir = os.path.dirname(file_path)
            hwp_file_name = os.path.basename(file_path)
            html_file_dir = os.path.join(hwp_file_dir, hwp_file_name.split('.')[0])
            html_file_path = os.path.join(html_file_dir, 'index.xhtml')
            hwp2html(file_path, html_file_dir)
            page = open(html_file_path, 'rt', encoding='utf-8').read()
            page = BeautifulSoup(page, 'html.parser')
            shutil.rmtree(html_file_dir)
            text = page.get_text(separator='\n')
            q_start = text.find('질의 요지')
            a_start = text.find('회답')
            r_start = text.find('이유')
            question = text[q_start+len('질의 요지'):a_start].strip() if q_start != -1 and a_start != -1 else ""
            answer = text[a_start+len('회답'):r_start].strip() if a_start != -1 and r_start != -1 else ""
            reason = text[r_start+len('이유'):].strip() if r_start != -1 else ""
            if '요청을 반려합니다' in answer or '요청을 반려합니다' in reason:
                print(f"❌ 제외: {file_path} (요청을 반려합니다 포함)")
                return None
            return clean_text(question), clean_text(answer), clean_text(reason)
            
        # NAL_1761 예외 처리
        if 'NAL_1761' in file_path:
            print(f"❌ 제외: {file_path} (신청인 철회)")
            return None

        hwp_file_dir = os.path.dirname(file_path)
        hwp_file_name = os.path.basename(file_path)
        html_file_dir = os.path.join(hwp_file_dir, hwp_file_name.split('.')[0])
        html_file_path = os.path.join(html_file_dir, 'index.xhtml')
        
        hwp2html(file_path, html_file_dir)
        
        page = open(html_file_path, 'rt', encoding='utf-8').read()
        page = BeautifulSoup(page, 'html.parser')
        shutil.rmtree(html_file_dir)

        tag_list = page.find_all('td')
        question, answer, reason = "", "", ""

        # 패턴: 띄어쓰기/개행 허용
        question_patterns = ['질의요지', '질의 요지', '요청대상행위', '요청 대상 행위']
        answer_patterns = ['회답', '판단']
        reason_patterns = ['이유', '판단이유', '판단 이유']

        for idx, tag in enumerate(tag_list):
            tag_text = tag.get_text().replace(" ", "").replace("\n", "")
            # 질의요지: 패턴 중 하나가 포함된 셀(제목)이면
            if any(pat.replace(" ", "") in tag_text for pat in question_patterns) and len(tag_text) < 12:
                question = tag_list[idx+1].get_text()
            # 회답
            if any(pat in tag_text for pat in answer_patterns) and len(tag_text) < 10:
                answer = tag_list[idx+1].get_text()
            # 이유
            if any(pat in tag_text for pat in reason_patterns) and len(tag_text) < 10:
                reason = tag_list[idx+1].get_text()

        # "요청을 반려합니다" 필터링
        if '요청을 반려합니다' in answer or '요청을 반려합니다' in reason:
            print(f"❌ 제외: {file_path} (요청을 반려합니다 포함)")
            return None

        return clean_text(question), clean_text(answer), clean_text(reason)

    def extract_from_hwpx(self, file_path: str) -> Optional[Tuple[str, str, str]]:
        """HWPX 파일에서 질의요지, 회답, 이유 추출"""
        # 특수 케이스 제외
        if 'LAW_3134' in file_path or 'NAL_2258' in file_path:
            print(f"❌ 제외: {file_path} (특수 케이스, 내용 무시)")
            return None

        # NAL_1761 예외 처리
        if 'NAL_1761' in file_path:
            print(f"❌ 제외: {file_path} (신청인 철회)")
            return None

        question, answer, reason = "", "", ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # HWPX 파일을 ZIP으로 압축해제
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # section XML 파일들 찾기
                contents_dir = os.path.join(temp_dir, 'Contents')
                section_files = [f for f in os.listdir(contents_dir) if f.startswith('section') and f.endswith('.xml')]
                
                all_text = ""
                cell_list = []

                for section_file in section_files:
                    section_path = os.path.join(contents_dir, section_file)
                    
                    with open(section_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    
                    # BeautifulSoup으로 XML 파싱
                    soup = BeautifulSoup(xml_content, 'lxml-xml')
                    
                    # 표 데이터 추출
                    tables = soup.find_all("hp:tbl")
                    for table in tables:
                        rows = table.find_all("hp:tr")
                        for row in rows:
                            cells = row.find_all("hp:tc")
                            for cell in cells:
                                cell_text = cell.get_text(strip=True)
                                if cell_text:
                                    cell_list.append(cell_text)
                                    all_text += cell_text + " "
                    
                    # 텍스트 추출
                    text_elements = soup.find_all(['hp:t', 'hp:tc'])
                    for elem in text_elements:
                        text = elem.get_text(strip=True)
                        if text:
                            all_text += text + " "
                
                # 셀 기반 section 추출
                for idx, cell in enumerate(cell_list):
                    cell_nospace = cell.replace(" ", "")
                    # 질의요지: '질의'가 포함된 셀(제목)이면 무조건
                    if '질의' in cell_nospace and len(cell_nospace) < 12 and idx+1 < len(cell_list):
                        question = cell_list[idx+1]
                    # 회답
                    if any(pat in cell_nospace for pat in ['회답', '판단']) and len(cell_nospace) < 10 and idx+1 < len(cell_list):
                        answer = cell_list[idx+1]
                    # 이유
                    if any(pat in cell_nospace for pat in ['이유', '판단이유', '판단이유']) and len(cell_nospace) < 10 and idx+1 < len(cell_list):
                        reason = cell_list[idx+1]

                # 만약 셀 기반 추출이 모두 실패하면 기존 패턴도 시도
                if not question and not answer and not reason:
                    text_parts = all_text.split()
                    text_joined = " ".join(text_parts)
                    patterns = {
                        'question': [
                            r'(질의요지|요청\s*대상\s*행위)[\s:：]*([\\s\\S]*?)(?=(회답|판단|이유|판단\s*이유|판단이유|$))'
                        ],
                        'answer': [
                            r'(회답|판단)[\s:：]*([\\s\\S]*?)(?=(이유|판단\s*이유|판단이유|$))'
                        ],
                        'reason': [
                            r'(이유|판단\s*이유|판단이유)[\s:：]*([\\s\\S]*?)(?=$)'
                        ]
                    }
                    def extract(patterns):
                        for pat in patterns:
                            m = re.search(pat, text_joined, re.IGNORECASE)
                            if m:
                                return m.group(2).strip()
                        return ""
                    question = extract(patterns['question'])
                    answer = extract(patterns['answer'])
                    reason = extract(patterns['reason'])
                
                # 패턴 매칭이 실패한 경우 간단한 분할 시도
                if not question and not answer and not reason:
                    text_length = len(all_text)
                    if text_length > 0:
                        third = text_length // 3
                        question = all_text[:third].strip()
                        answer = all_text[third:third*2].strip()
                        reason = all_text[third*2:].strip()
            
            except Exception as e:
                print(f"HWPX 파일 처리 중 오류: {e}")
                question = all_text if 'all_text' in locals() else ""
        
        # "요청을 반려합니다" 필터링
        if '요청을 반려합니다' in answer or '요청을 반려합니다' in reason:
            print(f"❌ 제외: {file_path} (요청을 반려합니다 포함)")
            return None

        return clean_text(question), clean_text(answer), clean_text(reason)

    def make_json_from_dir(self, target_dir: str, output_json: str) -> Dict[str, Any]:
        """디렉토리 내 문서들을 JSON으로 변환"""
        data = []
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "excluded_files": 0,
            "errors": []
        }
        
        if not os.path.exists(target_dir):
            print(f"❌ 디렉토리가 존재하지 않습니다: {target_dir}")
            return {"data": data, "stats": stats}
        
        files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
        stats["total_files"] = len(files)
        
        for file_name in files:
            file_path = os.path.join(target_dir, file_name)
            ext = pathlib.Path(file_path).suffix.lower()
            
            try:
                if ext == '.txt':
                    result = self.extract_from_txt(file_path)
                elif ext == '.hwp':
                    result = self.extract_from_hwp(file_path)
                elif ext == '.hwpx':
                    result = self.extract_from_hwpx(file_path)
                else:
                    print(f"⚠️  지원하지 않는 파일 형식: {file_name}")
                    continue

                # None 반환시 제외
                if result is None or all(x is None for x in result):
                    print(f"🚫 제외: {file_name} (필터 조건 불충족)")
                    stats["excluded_files"] += 1
                    continue

                question, answer, reason = result
                
                doc_id = os.path.splitext(file_name)[0]
                title = question if question else doc_id
                
                sections = []
                if question:
                    sections.append({"type": "질의요지", "text": question})
                if answer:
                    sections.append({"type": "회답", "text": answer})
                if reason:
                    sections.append({"type": "이유", "text": reason})
                
                item = {
                    "doc_id": doc_id,
                    "title": title,
                    "sections": sections,
                    "source": file_name,
                    "law_refs": [],
                    "tokens": len(question.split()) + len(answer.split()) + len(reason.split()),
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "quality_score": None,
                    "comments": [],
                    "status": "미검토"
                }
                
                data.append(item)
                stats["processed_files"] += 1
                print(f"✅ Success: {file_name}")
                
            except Exception as e:
                error_msg = f"❌ Error: {file_name} : {e}"
                print(error_msg)
                stats["errors"].append(error_msg)
        
        # JSON 파일 저장
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 완료! {len(data)}개 문서가 {output_json}에 저장되었습니다.")
        
        return {"data": data, "stats": stats}

    def update_document_quality(self, doc_id: str, quality_score: int, comments: List[str], 
                              data_file: str) -> bool:
        """문서 품질 정보 업데이트"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if item["doc_id"] == doc_id:
                    item["quality_score"] = quality_score
                    item["comments"] = comments
                    item["status"] = "검토완료" if quality_score >= 3 else "수정필요"
                    break
            else:
                return False
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ 품질 정보 업데이트 에러: {e}")
            return False