# 금융감독원 크롤링
"""
금융감독원 검사결과제재 전자금융 관련 문서 크롤러
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def is_electronic_finance_related( content: str = "") -> bool:
    """전자금융 관련 여부 판단"""
    # 전자금융 관련 키워드
    content_keywords = [
        '전자금융', '핀테크', '온라인', '모바일', '인터넷뱅킹', '전자지급',
        '전자결제', 'API', '오픈뱅킹', '마이데이터', '전자서명', '정보보호',
        '개인정보', '사이버', '해킹', '보안', '전산', 'IT', '시스템', '네트워크'
    ]
    
    # 내용 확인
    if content:
        for keyword in content_keywords:
            if keyword in content:
                return True
    
    return False


class FSSCrawler:
    """금융감독원 크롤러"""
    
    def __init__(self, base_dir: str = "../data/FSS_SANCTION"):
        self.base_url = "https://www.fss.or.kr"
        self.base_dir = Path(base_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.lock = threading.Lock()
        
    def download_file(self, file_url: str, filename: str, save_dir: Path) -> Optional[Path]:
        """파일 다운로드"""
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        
        if filepath.exists():
            print(f"📁 이미 다운로드됨: {filepath}")
            return filepath
        
        try:
            resp = self.session.get(file_url, timeout=20)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                with self.lock:
                    print(f"✅ 다운로드 완료: {filepath}")
                return filepath
            else:
                with self.lock:
                    print(f"❌ 파일 요청 실패: {resp.status_code}")
        except Exception as e:
            with self.lock:
                print(f"❌ 다운로드 중 에러 발생: {e}")
        return None
        
    def get_sanction_list(self, page: int = 1) -> Dict:
        """검사결과제재 목록 조회"""
        url = f"{self.base_url}/fss/job/openInfo/list.do"
        params = {
            'menuNo': '200476',
            'pageIndex': page,
            'sdate': '2014-01-01',
            'edate': datetime.now().strftime('%Y-%m-%d'),
            'searchCnd': '4',
            'searchWrd': ''
        }
        
        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"❌ 목록 조회 실패: 상태코드 {resp.status_code}")
                return {}
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 전체 건수 추출
            total_count = 0
            total_elem = soup.find(text=re.compile(r'총\s*\d+\s*건'))
            if total_elem:
                match = re.search(r'(\d+)', str(total_elem))
                if match:
                    total_count = int(match.group(1))
            
            # 테이블에서 데이터 추출
            items = []
            rows = soup.select('table tbody tr')
            
            for row in rows:
                tds = row.find_all('td')
                if len(tds) >= 6:
                    item = {
                        'no': tds[0].get_text(strip=True),
                        'institution': tds[1].get_text(strip=True),
                        'date': tds[2].get_text(strip=True),
                        'views': tds[5].get_text(strip=True)
                    }
                    
                    # 상세 링크 추출
                    link_tag = tds[3].find('a')
                    if link_tag:
                        href = link_tag.get('href')
                        if href:
                            # URL에서 파라미터 추출
                            parsed = urllib.parse.urlparse(href)
                            params = urllib.parse.parse_qs(parsed.query)
                            item['examMgmtNo'] = params.get('examMgmtNo', [''])[0]
                            item['emOpenSeq'] = params.get('emOpenSeq', [''])[0]
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
    
    def get_sanction_detail_and_download(self, item: Dict, save_dir: str = "data/FSS_SANCTION") -> Dict:
        """제재 상세 정보 조회 및 파일 다운로드"""
        detail_url = item.get('detail_url')
        if not detail_url:
            return item
            
        try:
            from date_normalizer import normalize_date as _normalize_date
            resp = self.session.get(detail_url, timeout=20)
            if resp.status_code != 200:
                print(f"❌ 상세 조회 실패: 상태코드 {resp.status_code}")
                return item
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 상세 정보 추출
            detail = item.copy()
            
            # 테이블에서 정보 추출
            for table in soup.find_all('table', class_='tbl_view'):
                for tr in table.find_all('tr'):
                    th = tr.find('th')
                    td = tr.find('td')
                    if th and td:
                        key = th.get_text(strip=True)
                        value = td.get_text(' ', strip=True)
                        
                        if key == '제재대상기관':
                            detail['institution'] = value
                        elif key == '제재조치요구일':
                            detail['date'] = value
                        elif key == '제재조치요구내용':
                            detail['content'] = value
            
            # 파일 다운로드
            save_path = self.base_dir / save_dir
            _normalized = _normalize_date(detail.get('date', ''))
            date_str = _normalized.replace('-', '') if _normalized else datetime.now().strftime('%Y%m%d')
            detail['date'] = _normalized or detail.get('date', '')
            institution = re.sub(r'[^\w\s-]', '', detail.get('institution', 'unknown'))
            institution = institution.replace(' ', '_')[:30]
            
            download_count = 0
            downloaded_files = []
            
            # 첨부파일 링크 찾기
            for a in soup.find_all('a', href=True):
                file_href = a['href']
                
                # 다운로드 링크 패턴 확인
                if '/fss.hpdownload' in file_href or 'download' in file_href.lower():
                    full_url = urllib.parse.urljoin(self.base_url, file_href)
                    
                    # 파일명 추출
                    name_tag = a.find('span', class_='name')
                    if name_tag:
                        filename = name_tag.get_text(strip=True)
                    else:
                        # URL에서 파일명 추출 시도
                        filename_match = re.search(r'file=([^&]+)', file_href)
                        if filename_match:
                            filename = urllib.parse.unquote(filename_match.group(1))
                        else:
                            filename = a.get_text(strip=True)
                    
                    # 파일명 정리
                    if not filename or filename == '내용보기':
                        continue
                        
                    # 확장자 확인
                    if not any(filename.lower().endswith(ext) for ext in ['.pdf', '.hwp', '.hwpx']):
                        # 확장자가 없으면 추가
                        if '.' not in filename:
                            filename += '.pdf'
                        else:
                            continue
                    
                    # 파일명 생성
                    safe_filename = f"SANCTION_{date_str}_{institution}_{detail.get('examMgmtNo', 'unknown')}_{download_count+1}_{filename}"
                    safe_filename = re.sub(r'[^\w\s.-]', '_', safe_filename)
                    
                    # 다운로드
                    filepath = self.download_file(full_url, safe_filename, save_path)
                    if filepath:
                        downloaded_files.append(str(filepath))
                        download_count += 1
                    
                    time.sleep(0.5)  # 서버 부하 방지 (단축)
            
            detail['downloaded_files'] = downloaded_files
            detail['download_count'] = download_count
            
            # 상세 정보를 JSON으로 저장
            json_filename = f"SANCTION_{date_str}_{institution}_{detail.get('examMgmtNo', 'unknown')}.json"
            json_path = save_path / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detail, f, ensure_ascii=False, indent=2)
            
            return detail
            
        except Exception as e:
            print(f"❌ 상세 조회 에러: {type(e).__name__} - {e}")
            return item
    
    def get_existing_sanctions(self, save_dir: str = "data/FSS_SANCTION") -> set:
        """기존 저장된 제재 정보 확인"""
        existing = set()
        save_path = self.base_dir / save_dir
        
        if save_path.exists():
            for file in save_path.glob("*.json"):
                if file.name.startswith("SANCTION_"):
                    # 파일명에서 examMgmtNo 추출
                    match = re.search(r'_(\d{9,})\.json', file.name)
                    if match:
                        existing.add(match.group(1))
        
        return existing
    
    def process_item_parallel(self, item: Dict, existing: set) -> Optional[Dict]:
        """개별 항목 처리 (병렬 처리용)"""
        # 이미 저장된 항목은 건너뛰기
        if item.get('examMgmtNo') and item.get('examMgmtNo') in existing:
            return None
        
        with self.lock:
            print(f"\n🔍 처리 중: {item['institution']} ({item['date']})")
        
        # 상세 정보 조회 및 다운로드
        detail = self.get_sanction_detail_and_download(item)
        
        if detail.get('download_count', 0) > 0:
            with self.lock:
                print(f"📥 {detail['download_count']}개 파일 다운로드 완료")
        else:
            with self.lock:
                print("📁 첨부파일 없음")
        
        file_info = {
            'type': '검사결과제재',
            'examMgmtNo': item.get('examMgmtNo'),
            'institution': detail.get('institution', ''),
            'date': detail.get('date', ''),
            'downloaded_files': detail.get('downloaded_files', []),
            'timestamp': datetime.now().isoformat()
        }
        
        return file_info
    
    def crawl_electronic_finance_sanctions(self) -> List[Dict]:
        """전자금융 관련 제재 정보 크롤링"""
        new_files = []
        
        print(f"=== 금융감독원 전자금융 제재 크롤링 시작: {datetime.now()} ===")
        
        try:
            # 기존 파일 확인
            existing = self.get_existing_sanctions()
            print(f"기존 제재 정보: {len(existing)}개")
            
            page = 1
            total_electronic = 0
            consecutive_empty = 0
            
            while True:
                print(f"\n📄 {page}페이지 조회 중...")
                result = self.get_sanction_list(page)
                
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
                
                # 병렬 처리로 항목들 처리
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_item = {
                        executor.submit(self.process_item_parallel, item, existing): item 
                        for item in items
                    }
                    
                    for future in as_completed(future_to_item):
                        try:
                            file_info = future.result()
                            if file_info:
                                new_files.append(file_info)
                                page_electronic += 1
                                total_electronic += 1
                        except Exception as e:
                            item = future_to_item[future]
                            print(f"❌ 항목 처리 중 에러 ({item.get('institution', 'unknown')}): {e}")
                
                time.sleep(0.5)  # 페이지 간 대기 시간 단축
                
                print(f"  → 이 페이지에서 {page_electronic}개 전자금융 제재 발견")
                
                # 다음 페이지 확인
                total_count = result.get('total_count', 0)
                if total_count > 0 and page * 10 >= total_count:
                    print(f"\n✅ 모든 페이지 조회 완료 (총 {total_count}건)")
                    break
                
                page += 1
                
                # 안전장치
                if page > 500:  # 최대 500페이지까지만
                    print("⚠️ 500페이지 초과, 안전 중단")
                    break
            
            print(f"\n=== 크롤링 완료 ===")
            print(f"🔵 새로운 전자금융 제재 정보: {len(new_files)}개")
            print(f"📊 전체 전자금융 제재: {total_electronic}개")
            
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
        log_file = log_dir / f"fss_sanctions_{today}.json"
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(new_files, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📋 크롤링 로그 저장: {log_file}")


if __name__ == "__main__":
    # 크롤러 인스턴스 생성
    crawler = FSSCrawler()
    
    # 전자금융 관련 제재 정보 크롤링
    new_files = crawler.crawl_electronic_finance_sanctions()
    
    # 크롤링 로그 저장
    crawler.save_crawl_log(new_files)
    
    print("\n✨ 금융감독원 전자금융 제재 크롤링 완료!")
