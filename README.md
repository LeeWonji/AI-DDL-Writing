# AI-DDL Writing

학년·출판사·단원을 선택한 뒤 학생 영어 작문에 대한 AI 피드백을 받는 웹 앱입니다.  
오류 추출, 핵심 표현 분류, 수정 제안, 단서 예문(KWIC)을 제공합니다.

## 실행 방법

1. 가상환경 생성 및 패키지 설치  
   `python -m venv .venv`  
   `.venv\Scripts\activate`  
   `pip install -r requirements.txt`

2. OpenAI API 키를 환경 변수로 설정 (PowerShell 예시)  
   `$env:OPENAI_API_KEY = "your-api-key"`

3. 서버 실행  
   `python "AI-DDL Writing.py"`  
   또는 **run.bat** 더블클릭

4. 브라우저에서 http://127.0.0.1:5001 접속

## GitHub에 올리기

프로젝트 폴더에서 아래를 **순서대로** 실행하세요.

```bash
cd "c:\Users\이원지\Desktop\CURSOR\AI-DDL Writing"

# 1) Git 사용자 설정 (한 번만 하면 됨)
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"

# 2) 저장소 초기화 및 첫 커밋 (이미 .gitignore는 만들어 둠)
git init
git add -A
git commit -m "Initial commit: AI-DDL Writing"

# 3) GitHub에서 새 저장소 생성
#    - https://github.com/new 에서 Repository name: AI-DDL-Writing (또는 AI-DDL-Writing)
#    - Create repository 후 나오는 URL 복사

# 4) 원격 저장소 연결 및 푸시
git remote add origin https://github.com/YOUR_USERNAME/AI-DDL-Writing.git
git branch -M main
git push -u origin main
```

`YOUR_USERNAME`을 본인 GitHub 아이디로 바꾸면 됩니다.
