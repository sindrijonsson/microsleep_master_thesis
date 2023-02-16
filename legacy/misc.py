import requests
from getpass import getpass
from io import StringIO
from bs4 import BeautifulSoup

def _get_usleep_token():
    
    def get_csrf(x):
        soup = BeautifulSoup(x.content, 'html.parser')
        info = soup.find('script', {'type':'text/javascript'})
        txt = info.get_text()
        start = txt.find('"')
        end = txt.find('"',start+1)
        csrf = txt[start+1:end]
        return csrf

    login_url = "https://sleep.ai.ku.dk/login"
    token_url = "https://sleep.ai.ku.dk/token/html"

    email = 's202056@student.dtu.dk'
    pwd = getpass("Password: ")

    _useless = '<p>The following personal access token is valid for 12 hours:</p><p class="codetext">'

    with requests.Session() as session:
        login_page = session.get(login_url)
        csrf = get_csrf(login_page)
        
        token_payload = myobj = {'email': email,
                             'password': pwd,
                             'csrf_token': csrf,
                             'login-submit': 'Log In'}
        
        login_post = session.post(login_url, data = token_payload)
        token_request = session.get(token_url)
        token_info = BeautifulSoup(token_request.content, "html.parser")
        token = str(token_info)[len(_useless):-4]
    
    return token
    