
class Region:
    def __init__(self, obj_id, obj_name, boundary, tti, speed):
        self.obj_id=obj_id
        self.obj_name=obj_name
        self.boundary=boundary
        self.tti=tti
        self.speed=speed

    def getCenter(self):
        x=0
        y=0
        count=0
        for i in self.boundary:
            for j in i:
                x+=j[0]
                y+=j[1]
                count+=1
        x/=count
        y/=count
        return x, y

class District(Region):
    pass

class Road(Region):
    pass