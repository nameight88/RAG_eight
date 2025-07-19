"""
금융감독원 경영유의 사항 등 공시 전자금융 관련 문서 크롤러
- 대상: https://www.fss.or.kr/fss/job/openInfoImpr/list.do
- 기간: 2014-01-01 ~ 현재
- 필터: 전자금융 관련 부서 및 키워드
"""

import requests
import time
import os
import json
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any


def is_electronic_finance_related(institution: str = "", content: str = "", filename: str = "") -> bool:
    """문서 내용에서 지정된 키워드 포함 여부 판단"""
    
    # 사용자 지정 키워드
    keywords = [
        '전자금융',
        '정보처리위탁',
        '정보처리 업무 위탁에 관한 규정',
        '신용정보법',
        '신용정보의 이용 및 보호에 관한 법률',
        '신용정보업감독규정',
        '신용정보업감독업무시행세칙'
    ]
    
    # 문서 내용에서 키워드 확인 (가장 중요)
    if content:
        for keyword in keywords:
            if keyword in content:
                return True
    
    # 파일명에서 키워드 확인
    if filename:
        for keyword in keywords:
            if keyword in filename:
                return True
    
    # 기관명에서 키워드 확인
    if institution:
        for keyword in keywords:
            if keyword in institution:
                return True
    
    return False


def is_likely_electronic_finance_related(institution: str = "", filename: str = "") -> bool:
    """다운로드 전 사전 필터링 - 기관명과 파일명으로만 판단"""
    
    # 전자금융 관련 키워드
    keywords = [
        '전자금융',
        '정보처리위탁',
        '정보처리 업무 위탁',
        '신용정보법',
        '신용정보',
        '신용정보업감독',
        '핀테크',
        '간편결제',
        '전자결제',
        '온라인',
        '디지털',
        'IT',
        '정보통신',
        '시스템',
        '데이터',
        '개인정보'
    ]
    
    # 파일명에서 키워드 확인
    if filename:
        filename_lower = filename.lower()
        for keyword in keywords:
            if keyword in filename_lower:
                return True
    
    # 기관명에서 키워드 확인
    if institution:
        institution_lower = institution.lower()
        for keyword in keywords:
            if keyword in institution_lower:
                return True
    
    return False


def extract_text_from_file(file_path: str) -> str:
    """파일에서 텍스트 추출"""
    try:
        if file_path.lower().endswith('.pdf'):
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        
        elif file_path.lower().endswith(('.hwp', '.hwpx')):
            # HWP 파일 처리 (간단한 방법)
            try:
                import subprocess
                result = subprocess.run(['hwp5txt', file_path], 
                                      capture_output=True, text=True, encoding='utf-8')
                return result.stdout
            except:
                return ""
        
        else:
            return ""
    except Exception as e:
        print(f"❌ 텍스트 추출 에러 {file_path}: {e}")
        return ""


class FSSManagementCrawler:
    """금융감독원 경영유의 공시 크롤러"""
    
    def __init__(self, base_dir: str = "../../"):
        self.base_url = "https://www.fss.or.kr"
        self.base_dir = Path(base_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    def download_file(self, file_url: str, filename: str, save_dir: Path) -> Optional[Path]:
        """파일 다운로드"""
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        
        if filepath.exists():
            print(f"📁 이미 다운로드됨: {filepath}")
            return filepath
        
        try:
            # 세션 사용하여 다운로드
            session = requests.Session()
            session.headers.update(self.headers)
            
            resp = session.get(file_url, timeout=30, stream=True)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ 다운로드 완료: {filepath}")
                return filepath
            else:
                print(f"❌ 파일 요청 실패: {resp.status_code}")
        except Exception as e:
            print(f"❌ 다운로드 중 에러 발생: {e}")
        return None
        
    def get_management_list(self, page: int = 1) -> Dict:
        """경영유의 공시 목록 조회"""
        url = f"{self.base_url}/fss/job/openInfoImpr/list.do"
        params = {
            'menuNo': '200483',
            'pageIndex': page,
            'sdate': '2014-01-01',
            'edate': datetime.now().strftime('%Y-%m-%d'),
            'searchCnd': '4',
            'searchWrd': ''
        }
        
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if resp.status_code != 200:
                print(f"❌ 목록 조회 실패: 상태코드 {resp.status_code}")
                return {}
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 전체 건수 추출
            total_count = 0
            total_elem = soup.find(string=re.compile(r'총\s*\d+\s*건'))
            if total_elem:
                match = re.search(r'(\d+)', str(total_elem))
                if match:
                    total_count = int(match.group(1))
            
            # 테이블에서 데이터 추출
            items = []
            
            # 여러 테이블 클래스 시도
            table = soup.find('table', class_='tbl_list')
            if not table:
                table = soup.find('table', class_='list')
            if not table:
                table = soup.find('table')
            
            if table:
                # tbody 찾기
                tbody = table.find('tbody')
                if not tbody:
                    tbody = table
                    
                rows = tbody.find_all('tr')
                
                for row in rows:
                    tds = row.find_all('td')
                    
                    if len(tds) >= 5:
                        item = {
                            'no': tds[0].get_text(strip=True),
                            'institution': tds[1].get_text(strip=True),
                            'date': tds[2].get_text(strip=True),
                        }
                        
                        # 상세 링크 추출
                        link_tag = tds[3].find('a')
                        if link_tag:
                            onclick = link_tag.get('onclick', '')
                            href = link_tag.get('href', '')
                            
                            if onclick:
                                # JavaScript 함수에서 파라미터 추출
                                match = re.search(r"fn_detail\('([^']+)','([^']+)'\)", onclick)
                                if match:
                                    item['mngmCntnNo'] = match.group(1)
                                    item['emOpenSeq'] = match.group(2)
                                    # 상세 URL 구성
                                    detail_params = {
                                        'menuNo': '200483',
                                        'mngmCntnNo': item['mngmCntnNo'],
                                        'emOpenSeq': item['emOpenSeq']
                                    }
                                    item['detail_url'] = f"{self.base_url}/fss/job/openInfoImpr/view.do?" + urllib.parse.urlencode(detail_params)
                            elif href:
                                # href에서 직접 파라미터 추출 (새로운 방식)
                                item['detail_url'] = urllib.parse.urljoin(self.base_url, href)
                        
                        items.append(item)
            
            return {
                'total_count': total_count,
                'items': items,
                'page': page
            }
            
        except Exception as e:
            print(f"❌ 목록 조회 에러 (페이지 {page}): {type(e).__name__} - {e}")
            return {}
    
    def get_management_detail_and_download(self, item: Dict, save_dir: str = "data/FSS_MANAGEMENT") -> Dict:
        """경영유의 파일 다운로드"""
        file_url = item.get('detail_url')
        if not file_url:
            return item
            
        try:
            # 파일 다운로드 준비
            save_path = self.base_dir / save_dir
            date_str = re.sub(r'[^\d]', '', item.get('date', ''))[:8] or datetime.now().strftime('%Y%m%d')
            institution = re.sub(r'[^\w\s-]', '', item.get('institution', 'unknown'))
            institution = institution.replace(' ', '_')[:30]
            
            # 파일명 추출 (URL에서)
            filename = "document.pdf"  # 기본값
            if 'file=' in file_url:
                import urllib.parse
                parsed_url = urllib.parse.urlparse(file_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if 'file' in query_params:
                    filename = query_params['file'][0]
                    filename = urllib.parse.unquote(filename)
            
            # 파일 확장자 확인
            if not filename.lower().endswith(('.pdf', '.hwp', '.hwpx')):
                if '.' not in filename:
                    filename += '.pdf'
            
            # 안전한 파일명 생성
            safe_filename = f"MGMT_{date_str}_{institution}_{filename}"
            safe_filename = re.sub(r'[^\w\s.-]', '_', safe_filename)
            
            # 파일 다운로드
            downloaded_files = []
            filepath = self.download_file(file_url, safe_filename, save_path)
            if filepath:
                downloaded_files.append(str(filepath))
                print(f"📥 다운로드 완료: {safe_filename}")
                
                # 기본 정보만 저장 (내용 분석은 나중에)
                detail = item.copy()
                detail['title'] = f"{institution} 경영유의사항"
                detail['downloaded_files'] = downloaded_files
                detail['download_count'] = len(downloaded_files)
                detail['file_url'] = file_url
                detail['filename'] = filename
                detail['safe_filename'] = safe_filename
                
                # 기본 JSON으로 저장
                json_filename = f"MGMT_{date_str}_{institution}.json"
                json_path = save_path / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(detail, f, ensure_ascii=False, indent=2)
                
                return detail
            else:
                print(f"❌ 파일 다운로드 실패: {safe_filename}")
                return item
            
        except Exception as e:
            print(f"❌ 처리 에러: {type(e).__name__} - {e}")
            return item
    
    def get_existing_managements(self, save_dir: str = "data/FSS_MANAGEMENT") -> set:
        """기존 저장된 경영유의 정보 확인"""
        existing = set()
        save_path = self.base_dir / save_dir
        
        if save_path.exists():
            for file in save_path.glob("*.json"):
                if file.name.startswith("MGMT_"):
                    # 파일명에서 mngmCntnNo 추출
                    match = re.search(r'_(\d{9,})\.json', file.name)
                    if match:
                        existing.add(match.group(1))
        
        return existing
    
    def crawl_electronic_finance_managements(self) -> List[Dict]:
        """전자금융 관련 경영유의 정보 크롤링"""
        new_files = []
        
        print(f"=== 금융감독원 전자금융 경영유의 크롤링 시작: {datetime.now()} ===")
        
        try:
            # 기존 파일 확인
            existing = self.get_existing_managements()
            print(f"기존 경영유의 정보: {len(existing)}개")
            
            page = 1
            total_electronic = 0
            consecutive_empty = 0
            
            while True:
                print(f"\n📄 {page}페이지 조회 중...")
                result = self.get_management_list(page)
                
                if not result or not result.get('items'):
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        print("📭 연속 3페이지 빈 결과 - 크롤링 종료")
                        break
                    page += 1
                    continue
                
                consecutive_empty = 0
                items = result['items']
                print(f"  - {len(items)}개 항목 발견")
                
                page_electronic = 0
                page_skipped = 0
                
                for item in items:
                    # 이미 저장된 항목은 건너뛰기
                    if item.get('mngmCntnNo') and item.get('mngmCntnNo') in existing:
                        continue
                    
                    institution = item.get('institution', '')
                    file_url = item.get('detail_url', '')
                    
                    # 파일명 추출
                    filename = ""
                    if 'file=' in file_url:
                        import urllib.parse
                        parsed_url = urllib.parse.urlparse(file_url)
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        if 'file' in query_params:
                            filename = query_params['file'][0]
                            filename = urllib.parse.unquote(filename)
                    
                    # 사전 필터링 - 전자금융 관련성 확인
                    if not is_likely_electronic_finance_related(institution, filename):
                        page_skipped += 1
                        if page <= 3:  # 처음 3페이지만 스키핑 정보 출력
                            print(f"⏭️  스키핑: {institution} | {filename}")
                        continue
                    
                    # 디버깅: 처음 3페이지만 다운로드 대상 항목 출력
                    if page <= 3:
                        print(f"📋 다운로드 대상: {institution} | {filename}")
                    
                    # 전자금융 관련 파일만 다운로드
                    detail = self.get_management_detail_and_download(item)
                    
                    if detail and detail.get('download_count', 0) > 0:
                        file_info = {
                            'type': '경영유의사항',
                            'institution': detail.get('institution', ''),
                            'date': detail.get('date', ''),
                            'title': detail.get('title', ''),
                            'filename': detail.get('filename', ''),
                            'safe_filename': detail.get('safe_filename', ''),
                            'file_url': detail.get('file_url', ''),
                            'downloaded_files': detail.get('downloaded_files', []),
                            'timestamp': datetime.now().isoformat()
                        }
                        new_files.append(file_info)
                        page_electronic += 1
                        total_electronic += 1
                    
                    time.sleep(2)  # 서버 부하 방지
                
                print(f"  → 이 페이지에서 {page_electronic}개 전자금융 경영유의 다운로드, {page_skipped}개 스키핑")
                
                # 다음 페이지 확인
                total_count = result.get('total_count', 0)
                if total_count > 0 and page * 10 >= total_count:
                    print(f"\n✅ 모든 페이지 조회 완료 (총 {total_count}건)")
                    break
                
                page += 1
                time.sleep(1)  # 페이지 간 대기
                
                # 안전장치 제거 - 모든 페이지 크롤링
                # if page > 3:  # 테스트용: 3페이지까지만
                #     print("⚠️ 테스트 모드: 3페이지까지만 크롤링")
                #     break
            
            
            print(f"\n=== 크롤링 완료 ===")
            print(f"🔵 다운로드된 경영유의 정보: {len(new_files)}개")
            print(f"📊 전체 다운로드: {total_electronic}개")
            
            return new_files
            
        except KeyboardInterrupt:
            print("\n❌ 사용자에 의해 크롤링이 중단되었습니다.")
            return new_files
        except Exception as e:
            print(f"\n❌ 크롤링 중 예기치 못한 에러: {type(e).__name__} - {e}")
            return new_files
    
    def save_crawl_log(self, new_files: List[Dict]) -> None:
        """크롤링 결과 로그 저장"""
        if not new_files:
            return
        
        log_dir = self.base_dir / "crawl_logs"
        log_dir.mkdir(exist_ok=True)
        
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"fss_managements_{today}.json"
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(new_files, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📋 크롤링 로그 저장: {log_file}")

    def create_json_corpus(self, new_files: List[Dict]) -> None:
        """다운로드된 모든 문서를 분석하여 관련 문서들로 JSON Corpus 생성"""
        if not new_files:
            return
        
        print(f"\n📄 {len(new_files)}개 다운로드된 문서 내용 분석 중...")
        
        corpus_dir = self.base_dir / "corpus"
        corpus_dir.mkdir(exist_ok=True)
        
        today = datetime.now().strftime("%Y%m%d")
        corpus_file = corpus_dir / f"fss_management_corpus_{today}.json"
        
        corpus_data = {
            "metadata": {
                "source": "금융감독원 경영유의사항 공시",
                "created_date": datetime.now().isoformat(),
                "total_downloaded": len(new_files),
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
            "documents": []
        }
        
        relevant_count = 0
        
        for i, file_info in enumerate(new_files):
            print(f"📖 분석 중 ({i+1}/{len(new_files)}): {file_info.get('institution', '')}")
            
            # 다운로드된 파일 경로
            downloaded_files = file_info.get('downloaded_files', [])
            if not downloaded_files:
                continue
                
            file_path = downloaded_files[0]
            
            try:
                # 파일 내용 추출
                file_content = extract_text_from_file(file_path)
                
                if not file_content:
                    print(f"⚠️  텍스트 추출 실패: {file_info.get('safe_filename', '')}")
                    continue
                
                # 키워드 포함 여부 확인
                is_relevant = is_electronic_finance_related(
                    file_info.get('institution', ''), 
                    file_content, 
                    file_info.get('filename', '')
                )
                
                if is_relevant:
                    print(f"✅ 관련 문서 발견: {file_info.get('institution', '')}")
                    
                    # 발견된 키워드 찾기
                    found_keywords = []
                    keywords = corpus_data["metadata"]["keywords"]
                    for keyword in keywords:
                        if keyword in file_content or keyword in file_info.get('filename', '') or keyword in file_info.get('institution', ''):
                            found_keywords.append(keyword)
                    
                    doc_data = {
                        "id": f"FSS_MGMT_{file_info.get('date', '').replace('-', '')}_{file_info.get('institution', '').replace(' ', '_')}",
                        "title": file_info.get('title', ''),
                        "institution": file_info.get('institution', ''),
                        "date": file_info.get('date', ''),
                        "filename": file_info.get('filename', ''),
                        "content": file_content,
                        "content_length": len(file_content),
                        "found_keywords": found_keywords,
                        "file_url": file_info.get('file_url', ''),
                        "processed_date": file_info.get('timestamp', '')
                    }
                    corpus_data["documents"].append(doc_data)
                    relevant_count += 1
                else:
                    print(f"⏭️  관련 없음: {file_info.get('institution', '')}")
                    
            except Exception as e:
                print(f"❌ 파일 분석 에러 {file_path}: {e}")
                continue
        
        # 메타데이터 업데이트
        corpus_data["metadata"]["total_relevant"] = relevant_count
        
        # Corpus 저장
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📚 JSON Corpus 생성 완료: {corpus_file}")
        print(f"📊 전체 다운로드: {len(new_files)}개")
        print(f"📊 관련 문서: {relevant_count}개")
        print(f"📊 Corpus에 포함: {len(corpus_data['documents'])}개")


if __name__ == "__main__":
    # 크롤러 인스턴스 생성
    crawler = FSSManagementCrawler()
    
    # 전자금융 관련 경영유의 정보 크롤링
    new_files = crawler.crawl_electronic_finance_managements()
    
    # 크롤링 로그 저장
    crawler.save_crawl_log(new_files)
    
    # JSON Corpus 생성
    crawler.create_json_corpus(new_files)
    
    print("\n✨ 금융감독원 전자금융 경영유의 크롤링 완료!")
