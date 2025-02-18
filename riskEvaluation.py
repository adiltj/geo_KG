import re

class riskEvaluation:
    def __init__(self,data):
        self.data=data

    def karst(self,data):
        grade = 0
        if (len(data) > 3):
            grade = 70
        if (data.find('无') != -1):
            grade = 0
        return grade

    def fault(self,data):
        grade = 0
        if (len(data) > 3):
            grade = 70
        if (data.find('无') != -1):
            grade = 0
        return grade

    def lithology(self,data):
        grade = 0
        SolubleRock = ['灰岩', '白云岩', '泥灰岩', '泥质岩']
        semi_SolubleRock = ['泥质粉砂岩']
        for i in SolubleRock:
            if (data.find(i) != -1):
                grade = 100
        for i in semi_SolubleRock:
            if (data.find(i) != -1):
                grade = 80
        return grade


    def surroundingWater(self,data):
        return 0


    def rockIntegrity(self,data):
        grade = 0
        if (data.find('破碎') != -1):
            grade = 80
        elif (data.find('极破碎') != -1):
            grade = 100
        elif (data.find('较破碎') != -1):
            grade = 50
        elif (data.find('完整') != -1):
            grade = 0
        elif (data.find('较完整') != -1):
            grade = 25
        return grade


    def attitude(self,data):
        # 125°∠43°
        if (data.find('∠') != -1):
            dip = data.split('∠')[1]
            dip = re.sub(u"([^\u0030-\u0039])", "", dip)
        else:
            dip = re.sub(u"([^\u0030-\u0039])", "", data)[-2:]
        dip = float(dip)
        if (dip > 25 and dip < 65):
            grade = 100
        elif (dip < 10):
            grade = 0
        else:
            grade = 50
        return grade


    def waterSeepage(self,data):
        grade = 0
        if (data.find('涌水') != -1):
            grade = 100
        elif (data.find('淋雨') != -1):
            grade = 80
        elif (data.find('点滴') != -1):
            grade = 60
        elif (data.find('渗水') != -1):
            grade = 50
        elif (data.find('潮湿') != -1):
            grade = 40
        elif (data.find('稍湿') != -1):
            grade = 20
        elif (data.find('干燥') != -1):
            grade = 0
        return grade


    def waterCondition(self,data):
        grade = 0
        if (data.find('不发育') != -1):
            grade = 0
        elif (data.find('较发育') != -1):
            grade = 50
        else:
            grade = 100
        return grade


    def rockGrade(self,data):
        grade = 50
        if (data.find('Ⅰ') != -1):
            grade = 0
        elif (data.find('Ⅱ') != -1):
            grade = 20
        elif (data.find('Ⅲ') != -1):
            grade = 40
        elif (data.find('Ⅳ') != -1):
            grade = 60
        elif (data.find('Ⅴ') != -1):
            grade = 80
        elif (data.find('Ⅵ') != -1):
            grade = 100
        return grade
    def calculate(self):
        g1 = self.karst(self.data['karst'])
        g2 = self.fault(self.data['fault'])
        g3 = self.lithology(self.data['lithology'])
        g4 = self.surroundingWater(self.data['surroundingWater'])
        g5 = self.rockIntegrity(self.data['rockIntegrity'])
        g6 = self.attitude(self.data['attitude'])
        g7 = self.waterSeepage(self.data['waterSeepage'])
        g8 = self.waterCondition(self.data['waterCondition'])
        g9 = self.rockGrade(self.data['actualRockGrade'])
        scores = [g1,g2, g3, g4, g5, g6, g7, g8, g9]
        grades=self.evaluation(scores)
        risk=sum(grades)
        level=self.riskLevel(risk)
        return scores,grades,risk,level

    def evaluation(self,grades):
        weights={'岩溶类':[0.07,0.02,0.07,0.02,0.02,0.07,0.24,0.25,0.24],

                 '断层类':[0.02,0.07,0.02,0.07,0.07,0.02,0.24,0.25,0.24],

                 '正常类':[0,0,0.1,0.1,0.1,0.1,0.2,0.2,0.2]}
        if(grades[0]!=0):
            list1 = [x * y for x, y in zip(grades, weights['岩溶类'])]
        if(grades[0]==0 and grades[1]!=0):
            list1 = [x * y for x, y in zip(grades, weights['断层类'])]
        else:
            list1 = [x * y for x, y in zip(grades, weights['岩溶类'])]
        return list1
    def riskLevel(self,risk):
        if (risk <= 20):
            level = "低度风险"
        if (risk > 20 and risk <= 40):
            level = "中度风险"
        if (risk > 40 and risk <= 60):
            level = "较高度风险"
        if (risk > 60 and risk <= 80):
            level = "高度风险"
        if (risk > 80 and risk <= 100):
            level = "极高度风险"
        return level
if __name__ == '__main__':
    data = {
        'karst':'',
        "fault": "",
        "lithology": "泥灰岩",
        "surroundingWater": '',
        "rockIntegrity": "岩体较破碎~破碎",
        "attitude": "产状：44°∠42°",
        "waterSeepage": "潮湿",
        "waterCondition": "地下水不发育",
        "actualRockGrade": "Ⅳ级",
    }
    risk=riskEvaluation(data)
    scores,grades,risk,level=risk.calculate()
    print(scores,grades,risk,level)