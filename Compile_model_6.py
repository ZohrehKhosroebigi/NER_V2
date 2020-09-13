from keras_contrib.layers import crf

class Compilemodel():
    def compilemodel(self,myobj):
        myobj.model_.compile(optimizer="adam", loss=myobj.crf.loss_function, metrics=[myobj.crf.accuracy])

        myobj.model_.summary()
        freport = open("logs/report.txt", "a", encoding="utf8")
        freport.write("model_.summary---------" + str(myobj.model_.summary()) + "\n")

        return myobj.model_
#41eb9622e772a4f4501f5d210b57311a978124a1