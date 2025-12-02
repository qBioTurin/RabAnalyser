library(readxl)
library(readr)
library(dplyr)


###### step 1:
FPW1_perni <- read_csv("~/ProvaIntera/Input_step1/FPW1.csv")
FPW1_tealdi <- read_excel("~/ProvaIntera/Output_step1/FPW1.xlsx")


FPW1_perni = FPW1_perni %>% select(-ID_image)
dim(FPW1_tealdi)
dim(FPW1_perni)

summary(FPW1_tealdi$Circ)

difftable = FPW1_perni[,-1] - FPW1_tealdi[,-1]
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))

sum(abs(difftable),na.rm = T)


#####

JK2_perni <- read_csv("~/ProvaIntera/Input_step1/JK2.csv")
JK2_tealdi <- read_excel("~/ProvaIntera/Output_step1/JK2.xlsx")


JK2_perni = JK2_perni %>% select(-ID_image)
dim(JK2_tealdi)
dim(JK2_perni)

summary(JK2_tealdi$Circ)

difftable = JK2_perni[,-1] - JK2_tealdi[,-1]
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))

sum(abs(difftable),na.rm = T)


#####

WK1_perni <- read_csv("~/ProvaIntera/Input_step1/WK1.csv")
WK1_tealdi <- read_excel("~/ProvaIntera/Output_step1/WK1.xlsx")

WK1_perni = WK1_perni %>% select(-ID_image)
dim(WK1_tealdi)
dim(WK1_perni)

difftable = WK1_perni[,-1] - WK1_tealdi[,-1]
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))

sum(abs(difftable),na.rm = T)


###### step 2:
WK1_2_perni <- read_csv("~/ProvaIntera/PerniceResults/WK1_step2.csv")
WK1_2_tealdi <- read_excel("~/ProvaIntera/Output_step2/WK1_KSRab5_WholeRef.xlsx")

dim(WK1_2_tealdi)
dim(WK1_2_perni)


difftable = WK1_2_perni - WK1_2_tealdi
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))


JK2_2_perni <- read_csv("~/ProvaIntera/PerniceResults/JK2_step2.csv")
JK2_2_tealdi <- read_excel("~/ProvaIntera/Output_step2/JK2_KSRab5_WholeRef.xlsx")

dim(JK2_2_tealdi)
dim(JK2_2_perni)

difftable = JK2_2_perni - JK2_2_tealdi
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))


FPW1_2_perni <- read_csv("~/ProvaIntera/PerniceResults/FPW1_step2.csv")
FPW1_2_tealdi <- read_excel("~/ProvaIntera/Output_step2/FPW1_KSRab5_WholeRef.xlsx")

dim(FPW1_2_tealdi)
dim(FPW1_2_perni)


difftable = (FPW1_2_perni - FPW1_2_tealdi)
difftable[abs(difftable) < 1e-10] = 0
summary(abs(difftable))



### step 3:

FPW1_2_perni$Class = "FPW1"
JK2_2_perni$Class = "JK2"
WK1_2_perni$Class = "WK1"
step2Data = rbind(FPW1_2_perni, rbind(JK2_2_perni, WK1_2_perni))

FPW1_2_tealdi$Class = "FPW1"
JK2_2_tealdi$Class = "JK2"
WK1_2_tealdi$Class = "WK1"
step2Datatealdi = rbind(FPW1_2_tealdi, rbind(JK2_2_tealdi, WK1_2_tealdi))

GlioCells_KSvaluesRab5WholeRef_V2 <- read_excel("~/ProvaIntera/Input_step3/GlioCells_KSvaluesRab5WholeRef_V2.xlsx")

summary(step2Data[,-12] -GlioCells_KSvaluesRab5WholeRef_V2[,-12])
summary(step2Datatealdi[,-12] -GlioCells_KSvaluesRab5WholeRef_V2[,-12])
summary(step2Data[,-12] -step2Datatealdi[,-12])



