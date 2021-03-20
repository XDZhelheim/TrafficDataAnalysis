from Region import District, Road
import re

class RegionFactory:
    def __init__(self):
        self.__instance=None

    def get_instance(self):
        if self.__instance==None:
            self.__instance=RegionFactory()
        return self.__instance

    def create_region(self, obj_id, obj_name, geom:str, tti, speed):
        p=re.compile(r'[(](.*)[)]', re.S)
        pmin=re.compile(r'[(](.*?)[)]', re.S)
        boundary=re.findall(p, geom)[0]
        if geom.startswith("POLYGON"):
            # boundary=re.findall(p, boundary)[0]
            # boundary=boundary.split(",")
            # for i in range(len(boundary)):
            #     boundary[i]=boundary[i].split(" ")
            #     boundary[i][0]=float(boundary[i][0])
            #     boundary[i][1]=float(boundary[i][1])
            #     boundary[i]=tuple(boundary[i])
            # return District(obj_id, obj_name, tuple(boundary), tti, speed)
            return
        elif geom.startswith("MULTILINESTRING"):
            boundary=re.findall(pmin, boundary)
            for i in range(len(boundary)):
                boundary[i]=boundary[i].split(",")
                for j in range(len(boundary[i])):
                    boundary[i][j]=boundary[i][j].split(" ")
                    boundary[i][j][0]=float(boundary[i][j][0])
                    boundary[i][j][1]=float(boundary[i][j][1])
                    boundary[i][j]=tuple(boundary[i][j])
                boundary[i]=tuple(boundary[i])
            return Road(obj_id, obj_name, tuple(boundary), tti, speed)
        return None