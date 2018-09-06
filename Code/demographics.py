def isSameGender(g):
    genders = g.split('-')
    if genders[0] == genders[1]:
        return 1
    else:
        return 0

def getAgeRelation(a):
    if (a < 0):
        return "hostYounger"
    elif (a > 0):
        return "hostOlder"
    else:
        return "sameAge"
