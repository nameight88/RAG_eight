"""
금융위원회 전자금융 관련 문서 크롤러
- 비조치의견서 (NAL)
- 법령해석 (LAW)
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


def is_electronic_finance_related(title: str) -> bool:
    """제목으로 전자금융 관련 여부 판단"""
    keywords = [
        '전자금융', '핀테크', '가상자산', '암호화폐', '비트코인', 
        '블록체인', '전자지급', '전자결제', '모바일결제', 'P2P',
        '크라우드펀딩', '로보어드바이저', '온라인투자', '디지털',
        '전자상거래', 'API', '오픈뱅킹', '마이데이터', '빅테크',
        '전자서명', 'DLT', '분산원장', 'CBDC', '스테이블코인', 
        '클라우드', '전산', '선불', '개인신용정보', '비밀번호',
        '신용정보', '재해복구', '정보처리'
    ]
    return any(keyword in title for keyword in keywords)


def extract_section_text(soup, label):
    """HTML에서 특정 섹션 텍스트 추출"""
    tag = soup.find(lambda tag: tag.name in ["strong", "th"] and label in tag.get_text())
    if tag:
        if tag.name == "th":
            tr = tag.find_parent("tr")
            if tr:
                td = tr.find("td")
                if td:
                    text = td.get_text(" ", strip=True)
                    if not text:
                        print(f"⚠️ [extract_section_text] <th>{label}</th> 내용이 비어있음!")
                    return text
        next_tag = tag.find_next(["div", "p", "td"])
        if next_tag:
            text = next_tag.get_text(" ", strip=True)
            if not text:
                print(f"⚠️ [extract_section_text] <strong>{label}</strong> 내용이 비어있음!")
            return text
    print(f"⚠️ [extract_section_text] '{label}' 섹션을 찾지 못함!")
    return ""


def parse_content(soup, base_url: str, idx: int, title: str, info: Dict) -> Dict:
    """웹페이지 내용 파싱"""
    summary = extract_section_text(soup, "질의요지")
    answer = extract_section_text(soup, "회답")
    reason = extract_section_text(soup, "이유")

    # 첨부파일 링크 추출
    hwp_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".hwp") or href.endswith(".hwpx"):
            file_url = href if href.startswith("http") else base_url + href
            hwp_links.append(file_url)

    return {
        "idx": idx,
        "title": title,
        "info": info,
        "summary": summary,
        "answer": answer,
        "reason": reason,
        "hwp_links": hwp_links
    }


class FSCCrawler:
    """금융위원회 크롤러"""
    
    def __init__(self, base_dir: str = "../../"):
        self.base_url = "https://better.fsc.go.kr"
        self.base_dir = Path(base_dir)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def parse_fsc_detail(self, idx: int, doc_type: str = "opinion", check_title_only: bool = False) -> Optional[Dict]:
        """개별 문서 파싱"""
        try:
            if doc_type == "opinion":
                detail_url = f"{self.base_url}/fsc_new/replyCase/OpinionDetail.do"
                data = {"muNo": 86, "stNo": 11, "opinionIdx": idx, "actCd": "R"}
                prefix = "NAL"
            elif doc_type == "lawreq":
                detail_url = f"{self.base_url}/fsc_new/replyCase/LawreqDetail.do"
                data = {"muNo": 85, "stNo": 11, "lawreqIdx": idx, "actCd": "R"}
                prefix = "LAW"
            else:
                raise ValueError("doc_type은 'opinion' 또는 'lawreq'만 가능합니다.")

            resp = requests.post(detail_url, data=data, headers=self.headers, timeout=10)
            
            if resp.status_code != 200:
                print(f"🌐 HTTP 에러 {doc_type} {idx}: 상태코드 {resp.status_code}")
                return None
                
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # 제목 추출
            title_tag = soup.select_one("table.tbl-view.two td.subject")
            if not title_tag:
                title_tag = soup.select_one("table.tbl-view td.subject")
            if not title_tag:
                title_tag = soup.select_one("table.tbl-write .subject")
            if not title_tag:
                title_tag = soup.select_one(".subject")
            if not title_tag:
                title_tag = soup.find("h3")

            title = title_tag.get_text(strip=True) if title_tag else ""
            
            if not title:
                return None
                
            if check_title_only:
                return {"title": title, "idx": idx}

            # 상세 정보 추출
            info = {}
            category = ""
            
            # 메타데이터 추출
            metadata = {}
            for row in soup.select("table.tbl-view tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td and not td.get("class") == ["subject"]:
                    key = th.get_text(strip=True)
                    val = td.get_text(" ", strip=True)
                    metadata[key] = val
                    
                    if key == "분야" or key == "분류":
                        category = val.strip()

            # 상세 내용 추출
            content_sections = {}
            for row in soup.select("table.tbl-write tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td and not td.get("class") == ["subject"]:
                    key = th.get_text(strip=True)
                    val = td.get_text(" ", strip=True)
                    content_sections[key] = val

            info.update(metadata)
            info.update(content_sections)

            result = parse_content(soup, self.base_url, idx, title, info)
            result["category"] = category
            return result
            
        except requests.exceptions.Timeout:
            print(f"⏰ 타임아웃 에러 {doc_type} {idx}: 10초 초과")
            return None
        except requests.exceptions.ConnectionError:
            print(f"🌐 연결 에러 {doc_type} {idx}: 네트워크 문제")
            return None
        except requests.exceptions.RequestException as e:
            print(f"🌐 요청 에러 {doc_type} {idx}: {e}")
            return None
        except Exception as e:
            print(f"❌ 파싱 에러 {doc_type} {idx}: {type(e).__name__} - {e}")
            return None

    def save_fsc_detail(self, result: Dict, save_dir: str = "data/FS_NAL", prefix: str = "NAL") -> Dict:
        """결과 저장"""
        save_path = self.base_dir / save_dir
        save_path.mkdir(exist_ok=True)
        
        idx = result["idx"]
        saved_path = None
        
        try:
            if result["hwp_links"]:
                for hwp_link in result["hwp_links"]:
                    try:
                        parsed_url = urllib.parse.urlparse(hwp_link)
                        org_file_name = urllib.parse.parse_qs(parsed_url.query).get('orgFileName', [f'{prefix}_{idx}.hwp'])[0]
                        ext = os.path.splitext(org_file_name)[-1] if '.' in org_file_name else '.hwp'
                        file_name = f'{prefix}_{idx}{ext}'
                        file_path = save_path / file_name
                        
                        if file_path.exists():
                            saved_path = str(file_path)
                            print(f"📄 이미 존재: {file_path}")
                            continue
                            
                        resp = requests.get(hwp_link, stream=True, headers=self.headers, timeout=30)
                        
                        if resp.status_code == 200:
                            with open(file_path, "wb") as f:
                                for chunk in resp.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            saved_path = str(file_path)
                            print(f"💾 다운로드 완료: {file_path}")
                            time.sleep(2)
                            break
                        else:
                            print(f"❌ 다운로드 실패 {prefix} {idx}: HTTP {resp.status_code}")
                            
                    except requests.exceptions.Timeout:
                        print(f"⏰ 다운로드 타임아웃 {prefix} {idx}: {hwp_link}")
                    except Exception as e:
                        print(f"❌ 다운로드 에러 {prefix} {idx}: {type(e).__name__} - {e}")
            else:
                # 텍스트로 저장
                file_name = f'{prefix}_{idx}.txt'
                file_path = save_path / file_name
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"제목: {result['title']}\n")
                    for k, v in result["info"].items():
                        f.write(f"{k}: {v}\n")
                saved_path = str(file_path)
                print(f"📝 텍스트 저장 완료: {file_path}")
            
            result["saved_path"] = saved_path
            return result
            
        except Exception as e:
            print(f"❌ 저장 에러 {prefix} {idx}: {type(e).__name__} - {e}")
            result["saved_path"] = None
            return result

    def get_electronic_finance_opinion_list(self) -> List[int]:
        """전자금융 비조치의견서 목록 추출"""
        url = f"{self.base_url}/fsc_new/replyCase/selectReplyCaseOpinionList.do"
        
        opinion_indices = []
        page = 0
        
        while True:
            try:
                data = {
                    'draw': page + 1,
                    'start': page * 10,
                    'length': 10,
                    'searchCategory': '4',  # 전자금융 분류코드
                    'searchKeyword': '',
                    'searchCondition': '',
                    'searchReplyRegDateStart': '',
                    'searchReplyRegDateEnd': '',
                    'searchStatus': '',
                    'searchLawType': '',
                    'searchChartIdx': ''
                }
                
                resp = requests.post(url, data=data, timeout=30)
                if resp.status_code != 200:
                    print(f"❌ 목록 조회 실패: 상태코드 {resp.status_code}")
                    break
                    
                json_data = resp.json()
                records = json_data.get('data', [])
                
                if not records:
                    break
                    
                for record in records:
                    opinion_idx = record.get('opinionIdx')
                    if opinion_idx:
                        opinion_indices.append(opinion_idx)
                        
                print(f"📄 전자금융 비조치의견서 {page+1}페이지: {len(records)}개 발견")
                page += 1
                
                if page > 1000:
                    print("⚠️ 1000페이지 초과, 안전 중단")
                    break
                    
            except Exception as e:
                print(f"❌ 목록 조회 에러 (페이지 {page+1}): {e}")
                break
                
        print(f"✅ 전자금융 비조치의견서 총 {len(opinion_indices)}개 발견")
        return sorted(opinion_indices)

    def get_electronic_finance_lawreq_list(self) -> List[int]:
        """법령해석 목록에서 전자금융 관련 제목만 추출"""
        url = f"{self.base_url}/fsc_new/replyCase/selectReplyCaseLawreqList.do"
        indices = []
        page = 0
        
        while True:
            data = {
                'draw': page + 1,
                'start': page * 10,
                'length': 10,
            }
            try:
                resp = requests.post(url, data=data, timeout=30)
                if resp.status_code != 200:
                    print(f"❌ 목록 조회 실패: 상태코드 {resp.status_code}")
                    break
                    
                records = resp.json().get('data', [])
                if not records:
                    break
                    
                for record in records:
                    title = record.get('title', '')
                    idx = record.get('lawreqIdx')
                    if idx and is_electronic_finance_related(title):
                        indices.append(idx)
                        
                print(f"📄 법령해석 {page+1}페이지: {len(records)}개 중 전자금융 {len([r for r in records if is_electronic_finance_related(r.get('title',''))])}개")
                page += 1
                
                if page > 1000:
                    print("⚠️ 1000페이지 초과, 안전 중단")
                    break
                    
            except Exception as e:
                print(f"❌ 목록 조회 에러 (페이지 {page+1}): {e}")
                break
                
        print(f"✅ 전자금융 법령해석 총 {len(indices)}개 발견")
        return sorted(indices)

    def get_existing_indices(self, save_dir: str, prefix: str) -> set:
        """기존 파일들의 인덱스 추출"""
        existing_indices = set()
        save_path = self.base_dir / save_dir
        
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
            return existing_indices
        
        for filename in save_path.iterdir():
            if filename.name.startswith(prefix):
                match = re.search(f'{prefix}_(\\d+)', filename.name)
                if match:
                    existing_indices.add(int(match.group(1)))
        
        return existing_indices

    def crawl_new_documents(self) -> List[Dict]:
        """새로운 문서들 크롤링"""
        new_files = []
        
        print(f"=== 크롤링 시작: {datetime.now()} ===")
        
        try:
            # 1. 비조치의견서 크롤링
            print("\n=== 비조치의견서 크롤링 (전자금융 분야만) ===")
            existing_nal = self.get_existing_indices("data/FS_NAL", "NAL")
            print(f"기존 비조치의견서 파일: {len(existing_nal)}개")
            
            electronic_finance_indices = self.get_electronic_finance_opinion_list()
            new_indices = [idx for idx in electronic_finance_indices if idx not in existing_nal]
            
            print(f"전자금융 비조치의견서 총 {len(electronic_finance_indices)}개 중 새로운 파일 {len(new_indices)}개 발견")
            
            nal_count = 0
            for current_idx in new_indices:
                try:
                    result = self.parse_fsc_detail(current_idx, "opinion")
                    if result and result.get("title"):
                        saved_result = self.save_fsc_detail(result, "data/FS_NAL", "NAL")
                        
                        file_info = {
                            "type": "비조치의견서",
                            "idx": current_idx,
                            "title": result["title"],
                            "category": "전자금융",
                            "file_path": saved_result.get("saved_path"),
                            "timestamp": datetime.now().isoformat()
                        }
                        new_files.append(file_info)
                        nal_count += 1
                        print(f"✅ 비조치의견서 {current_idx}: [전자금융] {result['title'][:40]}...")
                    else:
                        print(f"❌ 비조치의견서 {current_idx}: 상세 페이지 없음")
                        
                except Exception as e:
                    print(f"❌ 비조치의견서 {current_idx} 처리 에러: {type(e).__name__} - {e}")
            
            # 2. 법령해석 크롤링
            print("\n=== 법령해석 크롤링 (목록에서 제목 필터링) ===")
            existing_law = self.get_existing_indices("data/FS_LAW", "LAW")
            lawreq_indices = self.get_electronic_finance_lawreq_list()
            new_indices = [idx for idx in lawreq_indices if idx not in existing_law]
            
            print(f"전자금융 법령해석 총 {len(lawreq_indices)}개 중 새로운 파일 {len(new_indices)}개 발견")

            law_count = 0
            for current_idx in new_indices:
                try:
                    result = self.parse_fsc_detail(current_idx, "lawreq")
                    if result and result.get("title"):
                        saved_result = self.save_fsc_detail(result, "data/FS_LAW", "LAW")
                        file_info = {
                            "type": "법령해석",
                            "idx": current_idx,
                            "title": result["title"],
                            "file_path": saved_result.get("saved_path"),
                            "timestamp": datetime.now().isoformat()
                        }
                        new_files.append(file_info)
                        law_count += 1
                        print(f"✅ 법령해석 {current_idx}: {result['title'][:50]}...")
                    else:
                        print(f"❌ 법령해석 {current_idx}: 상세 페이지 없음")
                except Exception as e:
                    print(f"❌ 법령해석 {current_idx} 처리 에러: {type(e).__name__} - {e}")
                    
            # 결과 요약
            print(f"\n=== 크롤링 완료 ===")
            print(f"🔵 새로운 비조치의견서 (전자금융): {nal_count}개")
            print(f"🟢 새로운 법령해석 (전자금융): {law_count}개")
            print(f"📁 총 새 파일: {len(new_files)}개")
            
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
        log_file = log_dir / f"new_files_{today}.json"
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(new_files, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"크롤링 로그 저장: {log_file}")

if __name__ == "__main__":
    # 크롤러 인스턴스 생성
    crawler = FSCCrawler()
    
    # 새로운 문서 크롤링 실행
    new_files = crawler.crawl_new_documents()
    
    # 크롤링 로그 저장
    crawler.save_crawl_log(new_files)
    
    print("크롤링 완료!")
