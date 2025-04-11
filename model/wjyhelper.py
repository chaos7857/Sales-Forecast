
import json


class DataHelper:
    def __init__(self, json_file="datawjy/comment_conclusion.json"):
        """
        初始化 CarDataset，可选从 JSON 文件加载数据
        :param json_file: JSON 文件路径或文件对象（可选）
        """
        self.data = None
        if json_file is not None:
            self.load_data(json_file)

    def load_data(self, json_file):
        """
        从文件路径或文件对象加载 JSON 数据
        :param json_file: 文件路径（str）或文件对象
        """
        try:
            if isinstance(json_file, str):  # 输入是文件路径
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:  # 输入是文件对象
                self.data = json.load(json_file)
        except FileNotFoundError:
            raise ValueError(f"错误：文件 {json_file} 不存在！")
        except json.JSONDecodeError:
            raise ValueError("错误：JSON 格式无效！")


    '''小米su7'''
    def get_XiaoMiSu7(self,year_month=None):#小米su7
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_xiaoMisu7={}
            for car in self.data:
                if car.get('car')=='小米su7':
                    key=car.get("time")
                    a_xiaoMisu7[key]=car.get("comment")
            return a_xiaoMisu7
        else:#有年月输入，输出评论list
            b_xiaoMisu7 = []
            for car in self.data:
                if car.get("car")=="小米su7" and car.get("time").startswith(year_month):
                    b_xiaoMisu7.append(car.get("comment"))
            return b_xiaoMisu7

    def get_XiaoMIsu7_all_comments(self):
        c_xiaoMisu7=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_XiaoMiSu7("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，小米 SU7 数据为：{}".format(i, j, su7_data))
                c_xiaoMisu7.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return c_xiaoMisu7


    '''Model Y'''
    def get_ModelY(self,year_month=None):#小米su7
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_ModelY={}
            for car in self.data:
                if car.get('car')=='Model Y':
                    key=car.get("time")
                    a_ModelY[key]=car.get("comment")
            return a_ModelY
        else:#有年月输入，输出评论list
            b_ModelY = []
            for car in self.data:
                if car.get("car")=="Model Y" and car.get("time").startswith(year_month):
                    b_ModelY.append(car.get("comment"))
            return b_ModelY

    def get_ModelY_all_comments(self):
        C_ModelY=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_ModelY("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，Model Y数据为：{}".format(i, j, su7_data))
                C_ModelY.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return C_ModelY

    '''Model 3'''
    def get_Model3(self,year_month=None):#小米su7
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_Model3={}
            for car in self.data:
                if car.get('car')=='小米su7':
                    key=car.get("time")
                    a_Model3[key]=car.get("comment")
            return a_Model3
        else:#有年月输入，输出评论list
            b_Model3 = []
            for car in self.data:
                if car.get("car")=="小米su7" and car.get("time").startswith(year_month):
                    b_Model3.append(car.get("comment"))
            return b_Model3

    def get_Model3_all_comments(self):
        c_Model3=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_Model3("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，Model3数据为：{}".format(i, j, su7_data))
                c_Model3.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return c_Model3

    '''比亚迪_汉'''
    def get_BYD_H(self,year_month=None):#小米su7
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_BYD_H={}
            for car in self.data:
                if car.get('car')=='比亚迪_汉':#根据实际需求来
                    key=car.get("time")
                    a_BYD_H[key]=car.get("comment")
            return a_BYD_H
        else:#有年月输入，输出评论list
            b_BYD_H = []
            for car in self.data:
                if car.get("car")=="比亚迪_汉" and car.get("time").startswith(year_month):
                    b_BYD_H.append(car.get("comment"))
            return b_BYD_H

    def get_BYD_H_all_comments(self):
        c_BYD_H=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_BYD_H("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，比亚迪_汉 数据为：{}".format(i, j, su7_data))
                c_BYD_H.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return c_BYD_H

    '''问界M5'''
    def get_WJ_M5(self,year_month=None):
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_WJ_M5={}
            for car in self.data:
                if car.get('car')=='问界M5':
                    key=car.get("time")
                    a_WJ_M5[key]=car.get("comment")
            return a_WJ_M5
        else:#有年月输入，输出评论list
            b_WJ_M5 = []
            for car in self.data:
                if car.get("car")=="问界M5" and car.get("time").startswith(year_month):
                    b_WJ_M5.append(car.get("comment"))
            return b_WJ_M5

    def get_WJ_M5_all_comments(self):
        c_WJ_M5=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_WJ_M5("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，问界M5 数据为：{}".format(i, j, su7_data))
                c_WJ_M5.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return c_WJ_M5

    '''极氪007'''
    def get_JK_007(self,year_month=None):
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_JK_007={}
            for car in self.data:
                if car.get('car')=='极氪007':
                    key=car.get("time")
                    a_JK_007[key]=car.get("comment")
            return a_JK_007
        else:#有年月输入，输出评论list
            b_JK_007 = []
            for car in self.data:
                if car.get("car")=="极氪007" and car.get("time").startswith(year_month):
                    b_JK_007.append(car.get("comment"))
            return b_JK_007

    def get_JK_007_all_comments(self):
        c_JK_007=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_JK_007("202" + str(i) + "-" + str(f"{j:02d}"))
                # print("第202{}年{:02d}月，极氪007 数据为：{}".format(i, j, su7_data))
                c_JK_007.append({"202" + str(i) + "-" + str(f"{j:02d}"):su7_data})
        return c_JK_007

    '''ZEEKR'''
    def get_ZEEKR(self,year_month=None):
        if self.data is None:
            raise ValueError("错误：未加载数据，请先调用 load_data()！")

        if year_month is None:#无输入年月，输出“时间：评论”dict
            a_ZEEKR={}
            for car in self.data:
                if car.get('car')=='ZEEKR':
                    key=car.get("time")
                    a_ZEEKR[key]=car.get("comment")
            return a_ZEEKR
        else:#有年月输入，输出评论list
            b_ZEEKR = []
            for car in self.data:
                if car.get("car")=="ZEEKR" and car.get("time").startswith(year_month):
                    b_ZEEKR.append(car.get("comment"))
            return b_ZEEKR

    def get_ZEEKR_all_comments(self):
        C_ZEEKR=[]
        for i in range(2, 6):
            for j in range(1, 13):
                su7_data = self.get_ZEEKR("202" + str(i) + "-" + str(f"{j:02d}"))
                print("第202{}年{:02d}月，ZEEKR 数据为：{}".format(i, j, su7_data))
                C_ZEEKR.append({"202" + str(i) + "-0" + str(j):su7_data})
        return C_ZEEKR




# ==================== 使用示例 ====================
if __name__ == "__main__":
    dataset = DataHelper()
    data=dataset.get_BYD_H_all_comments()
    for elem in data:
        for key in elem:
            print(key,len(elem[key]))
    # print("the all comments is",data_JK)




