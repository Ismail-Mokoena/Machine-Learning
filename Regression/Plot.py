import matplotlib.pyplot as plt


class Graph():
    def __init__(self, 
                 x:float, 
                 y:float,
                 pred:float, 
                 c:str,
                 c2:str,
                 title: str, 
                 x_label:str,
                 y_label:str) -> None:
        self.x = x
        self.y = y
        self.pred = pred
        self.c = c
        self.c2 = c2
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
    
    def show(self)->None:
        plt.scatter(self.x,self.y,color=self.c)
        # regresion line ~ predicted
        plt.plot(self.x,self.pred,color=self.c2)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.savefig("Fig_1")
        plt.show()  
        
    