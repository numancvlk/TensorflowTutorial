import tensorflow as tf
import numpy as np

#----------------------CREATING TENSORS----------------------

stringTensor = tf.Variable("this is a string tensor",tf.string) #string tensor
print(stringTensor)

intTensor = tf.Variable(120,tf.int32) #int tensor
print(intTensor)

floatTensor = tf.Variable(120.0,tf.float32) #float tensor
print(floatTensor)

stringConstant = tf.constant("this is a constant string",tf.string) #BUDA VARIABLE GİBİ AMA BUNU DEĞİŞTİREMEZSİN
print(stringConstant)

#----------------------TENSORLERI NORMALE ÇEVİRMEK----------------------

strT = stringTensor.numpy() 
print(strT)

intT = intTensor.numpy()
print(intT)

floatT = floatTensor.numpy()
print(floatT)

#----------------------TENSOR RANKLARINI ÖĞRENMEK----------------------

#KÖŞELİ PARANTEZE GÖRE ARTIYOR İŞTE
arrayConstant1 = tf.constant(5)
print(f"Rank -> {tf.rank(arrayConstant1).numpy()}") #0 RANK

arrayConstant2 = tf.constant([1,2,3])
print(f"Rank -> {tf.rank(arrayConstant2).numpy()}") #1 RANK

arrayConstant3 = tf.constant([[1,2,3]])
print(f"Rank -> {tf.rank(arrayConstant3).numpy()}") #2 RANK

arrayConstant4 = tf.constant([[[1,2,3]]])
print(f"Rank -> {tf.rank(arrayConstant4).numpy()}") #3 RANK

#----------------------CHANGING SHAPE----------------------

x = tf.ones([1,2,3]) # TOPLAM 6 ELEMAN YANİ RANKI 3, 2 SATIR 3 SÜTUNDAN OLUSUYOR
print(x)

y = tf.reshape(x,[2,3,1]) #RANKI 3, 3 SATIR 1 SÜTUNDAN OLUSUYOR bu 3 ünün çarpımı eleman sayısına eşit olmalı
print(y)

z = tf.reshape(x,[3,1,2]) #3 BLOĞA AYIR HER BLOKTA 3 SATIR 1 SÜTUN OLSUN DEDİK
print(z)

#----------------------SLICING TENSORS----------------------

matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) #TENSOR OLUSTURUYORUZ
print(tf.rank(tensor)) #TENSOR RANKI
print(tensor.shape) #TENSOR BOYUTU

select1 = tensor[1,2] #TEK BİR ELEMAN SEÇMEK
print(select1)

select2 = tensor[0] #TEK SATIR SEÇMEK
print(select2)

select3 = tensor[:,0] #TEK SUTUN SEÇMEK
print(select3)

select4 = tensor[1::2] #1. satırdan başla, her 2 adımda bir al
print(select4)
