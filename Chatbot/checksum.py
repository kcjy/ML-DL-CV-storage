import re
import datetime

def nric_checksum(idNumber):
    singaporeanIDMap = {0: "J",1: "Z", 2: "I", 3: "H", 4: "G", 5: "F", 6: "E", 7: "D", \
                         8: "C", 9: "B", 10: "A"}
    foreignerIDMap = {0: "X", 1: "W", 2: "U", 3: "T", 4: "R", 5: "Q", 6: "P", 7: "N", \
                         8: "M", 9: "L", 10: "K"}

    if len(idNumber) == 9:
        ints = re.search(r'\d+', idNumber).group()
        ints = [int(d) for d in ints]

        if (len(ints) == 7) == True:
            sum = ints[0]*2 + ints[1]*7 + ints[2]*6 + ints[3]*5 + ints[4]*4 + ints[5]*3 + ints[6]*2
            if (idNumber[0].upper() == ("G" or "F")) == True:
                sum = sum + 4
                remainder = sum%11
                last_letter = idNumber[8].upper()
                return(foreignerIDMap.get(remainder) == last_letter)
        
            elif (idNumber[0].upper() == ("S" or "T")) == True:
                remainder = sum%11
                last_letter = idNumber[8].upper()
                return(singaporeanIDMap.get(remainder) == last_letter)
            else:
                return(False)
        else:
            return(False)
    else:
        return(False)


def email_check(email):
    if ("@" and ".com") in email:
        return("Valid")
    else:
        return("Invalid")
        
def hp_check(number):
    if number.startswith("8") == True or \
    number.startswith("9") == True and \
    len(number) == 8:
        return "Valid"
    else:
        return "Invalid"

def dateerror_check(x):
    try:
        date = datetime.datetime.strptime(x, "%Y-%m-%d")
        if isinstance(date, datetime.datetime):
            return(True)
        else:
            return("False1")
    except ValueError:
        return("False2")