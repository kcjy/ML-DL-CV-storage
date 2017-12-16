#!/usr/bin/env python

from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import datetime
from checksum import nric_checksum, email_check, hp_check, dateerror_check

import json
import os
import re

from flask import Flask
from flask import request
from flask import make_response

# Flask app should start in global layout
app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processRequest(req):
    if req.get("result").get("action") == "getPhoneNumber":
        parameters = req.get("result").get("parameters")
        hp_no = parameters.get("PhoneNumber")
        if hp_check(hp_no) == "Valid":
            speech = "Your phone number is " + hp_no + ". Please provide your email address."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_phonenumber-followup", "lifespan":1, "parameters":{}}]
        elif hp_check(hp_no) == "Invalid":
            speech = "The phone number you have entered is not valid." 
            context = [{"name":"get_name-followup", "lifespan":1, "parameters":{}}]

    if req.get("result").get("action") == "getEmail":
        email = req.get("result").get("resolvedQuery")
        if email_check(email) == "Valid":
            speech = "Your email is " + email + ". Please give us your citizenship status, whether you \
            are a Singapore Citizen, PR or Employment Pass Holder."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_email-followup", "lifespan":1, "parameters":{}}]
        else:
            speech = "Please enter a valid email."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_phonenumber-followup", "lifespan":1, "parameters":{}}]

    if req.get("result").get("action") == "getCitizenship":
        citizenship = req.get("result").get("parameters").get("Citizenship")
        keyword_list = ["PR", "Permanent Resident", "Citizen", "Singaporean", "Employment Pass", "EP"]
        if any(item in citizenship for item in keyword_list):
            speech = "You are a " + citizenship + ". Please give me your NRIC number."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_citizenship-followup", "lifespan":1, "parameters":{}}]
        else:
            speech = "Please give me a valid Citizenship status."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_email-followup", "lifespan":1, "parameters":{}}]

    if req.get("result").get("action") == "getNRIC":
        parameters = req.get("result").get("parameters")
        nric = parameters.get("NRICnum")
        if len(nric) == 9 and nric[0].isalpha() == True and nric[8].isalpha() == True:
            if nric_checksum(nric) == True:
                speech = "Your NRIC number is " + nric + ". Please provide your Date of Birth."
                context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
                {"name":"get_nric-followup", "lifespan":1, "parameters":{}}]
            else:
                speech = "Please enter a valid NRIC(1)."
                context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
                {"name":"get_citizenship-followup", "lifespan":1, "parameters":{}}]
        else:
            speech = "Please enter a valid NRIC(2)."
            context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, \
            {"name":"get_citizenship-followup", "lifespan":1, "parameters":{}}]

    if req.get("result").get("action") == "getDOB":
        dob = req.get("result").get("parameters").get("DOB")
        if dateerror_check(dob) == True:
        	date = datetime.datetime.strptime(dob, "%Y-%m-%d")
        else:
        	speech = "Please enter a valid Date of Birth."
        	context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, {"name":"get_nric-followup", "lifespan":1, "parameters":{}}]

        if ((datetime.datetime.today() - date).days < (365*21)) == True:
        	speech = "Unfortunately, you are under the legal age limit (21) to use Hektor."
        	context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, {"name":"get_nric-followup", "lifespan":1, "parameters":{}}]
        else:
        	speech = "Your DOB is " + dob + ". Please provide the date of expiry as stated on your NRIC."
        	context = [{"name":"get_name-followup", "lifespan":5, "parameters":{}}, {"name":"get_dob-followup", "lifespan":1, "parameters":{}}]

    if req.get("result").get("action") == "getExpiryDate":
        date_of_exp = req.get("result").get("parameters").get("ExpiryDate")
        if dateerror_check(date_of_exp) == True:
        	date = datetime.datetime.strptime(date_of_exp, "%Y-%m-%d")
        else:
        	speech = "Please enter a valid Date of Expiry."
        	context = [{"name":"get_name-followup", "lifespan":1, "parameters":{}}, {"name":"get_dob-followup", "lifespan":1, "parameters":{}}]

        if (((date - datetime.datetime.today()).days) <= 0) == True:
            speech = "The Expiry Date that you have given is invalid."
            context = [{"name":"get_name-followup", "lifespan":1, "parameters":{}}, {"name":"get_dob-followup", "lifespan":1, "parameters":{}}]
        else:    
            speech = "Your date of expiry is " + date_of_exp + ". Thank you for completing the sign-up process."
            context = [{"name":"get_name-followup", "lifespan":1, "parameters":{}}, {"name":"get_expirydate-followup", "lifespan":1, "parameters":{}}]

        for item in req.get("result").get("contexts"):
            if item.get("name") == "get_name-followup":
                parameters = item.get("parameters")
                break
            else:
                parameters = None

        first_name = parameters.get("FirstName")
        last_name = parameters.get("LastName")
        hp_no = parameters.get("PhoneNumber")
        email_add = parameters.get("Email")
        nric = parameters.get("NRICnum")
        dob = parameters.get("DOB")

        #Sending HTTP Post request to Hektor Server
        #data = {'first_name': first_name, 
        #        'last_name': last_name, 
        #        'phone_number': hp_no, 
        #        'email_address': email_add, 
        #        'ic_number': nric, 
        #        'date_of_birth': dob}
        #response = requests.post('http://api.hektor.com.sg/webhook_signup', data)
        #Check for Flask documentation for POST request response. Check for 200.

    print("Response:")
    print(speech)
    print(context)

    return {
        "speech": speech,
        "displayText": speech,
        # "data": data,
        "contextOut": context,
        "source": "flask-hektorbot"
    }

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
