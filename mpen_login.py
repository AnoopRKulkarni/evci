#
# MPEN EVCI Tool Login Page
# Authors: Anoop R Kulkarni
# Version 4.0
# Mar 14, 2022

from flask import Flask, redirect, url_for, request, abort, make_response
import json
import jwt
import requests
import time

FLASK_APP = 'localhost:8080'
STL_APP = 'http://localhost:8501'
VERIFY_JWT = 'https://sso.mpensystems.com/verifytoken?ssoToken='
SSO = 'https://sso.mpensystems.com/login?serviceURL=http://'+FLASK_APP+'/callback_url'
HDRS = {'Authorization': 'Bearer MPEN-l1Q7zkOL59cRqWBkQ12ZiGVW2DBL'}

app = Flask(__name__)

public_key = b"-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1haecO9shdFPg0hfndH5vAJfu7rrmQizIxCrY15rIep4b/Ae+giQFo8H8MrTsuKlHAfx5Tb0yH7QR1k5VfTfBadI+XQUZR1gOdqqm47Qd/XlZbF6YyzaAiwvh9PM12UjyEud6+XQOEkiD9Nh8RnZzHcR+15d+j1Rcx36KC2TismBe5Y/LtGtUNkFoLfNokLPt44dOWlV5PeaSDZD6fTMVlH2GC09oFGDR4PYFNA0fdPZh5cw5WLDyssItGbFMLjxPnw+IwvkT9jQedlwIrWIo1fAnHAEIzbfOsAXKjK+cwjw4gGAlWznoaQdvNrxoOUy9lOjpdQmogK/Ws9P2h538wIDAQAB\n-----END PUBLIC KEY-----"

def setcookie(redirect_url,val):
   resp = make_response(redirect(redirect_url))
   resp.set_cookie('mpen_evci_user',val)
   return resp

@app.route('/')
def index():
   return redirect(SSO)
   
@app.route('/callback_url',methods = ['GET'])
def callback_url():
   decoded_jwt = {}
   args = request.args
   ssoToken = args['ssoToken']
   resp = requests.get(VERIFY_JWT + ssoToken, headers=HDRS)
   if resp.status_code == 200:
      encoded_jwt = resp.json()['token']
      decoded_jwt = jwt.decode(encoded_jwt, public_key, algorithms="RS256")
      email = decoded_jwt['email']
      if decoded_jwt:
         # Store cookie and check for loggedin status in mpen_evci.py
         return setcookie(STL_APP,email)
   else:
      return

if __name__ == "__main__":
   app.run(port = 8080)
