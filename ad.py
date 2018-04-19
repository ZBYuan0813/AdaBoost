#coding: utf-8
from numpy import *
def loadData(filename):
	dataMat=[]
	labelMat=[]
	numFeat=len(open(filename).readline().split('\t'))
	for line in open(filename).readlines():
		lineArr=[]
		curLine=line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(curLine[i])
		dataMat.append(lineArr)
		labelMat.append(curLine[numFeat-1])
	return dataMat,labelMat
#通过阈值进行分类，构造单层决策树
#单层决策树只能根据一个特征划分
def stumpClassify(dataMat,dimen,threshvalue,threshIneq):
	#先将标签设置为1
	print 'stump'
	print dataMat.shape[0]
	retArray=ones((dataMat.shape[0],1))
	if threshIneq=='lt':
		retArray[dataMat[:,dimen]<=threshvalue]=-1.0
	else:
		retArray[dataMat[:,dimen]>threshvalue]=-1.0
	return retArray

#从加权数据中寻找最低的分类错误率
def buildStump(dataMat,labelMat,weight):
	#将数据转换为矩阵容易操作
	dataMatrix=mat(dataMat,dtype(float))
	labelMatrix=mat(labelMat,dtype(float))
	w,h=labelMatrix.shape
	labelMatrix=labelMatrix.reshape((h,w))
	m,n=shape(dataMatrix)
	numStep=10.0
	bestStump={}
	bestClassEst=mat(zeros((m,1)))
	minError=inf 
	#对列属性进行计算，来寻找最好的特征
	for i in range(n):
		#print dataMatrix[:,i]
		rangeMin=dataMatrix[:,i].min()
		rangeMax=dataMatrix[:,i].max()
		stepSize=(rangeMax - rangeMin)/numStep

		#对每一列特征进行迭代，得到最好的分类结果，并记录结果
		#并且保存每一个弱分类器bestStump{}的参数
		for j in range(-1,int(numStep)+1):
			print 'j: {}'.format(j)
			for inequal in ['lt','gt']:
				threshvalue=rangeMin+float(j)*stepSize
				predictdVals=stumpClassify(dataMatrix,i,threshvalue,inequal)
				print 'predictdVals'
				print predictdVals.shape
				print labelMatrix.shape
				errArr=mat(ones((m,1)))
				errArr[predictdVals==labelMatrix]=0
				weightError=weight.T*errArr
				if weightError<minError:
					minError=weightError
					bestClassEst=predictdVals.copy()
					bestStump['dim']=i
					bestStump['thread']=threshvalue
					bestStump['ineq']=inequal
	return bestStump,minError,bestClassEst

#基于单层决策树的AdaBoost训练函数
#numIt指迭代次数 默认为40 当训练错误率达到0就会提前结束训练
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = []   #用于存储每次训练得到的弱分类器以及其输出结果的权重
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)  #数据集权重初始化为1/m
	aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值
	for i in range(numIt):
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)#在加权数据集里面寻找最低错误率的单层决策树
		#print "D: ",D.T
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#根据错误率计算出本次单层决策树输出结果的权重 max(error,1e-16)则是为了确保error为0时不会出现除0溢出
		bestStump['alpha'] = alpha#记录权重
		weakClassArr.append(bestStump)
		#print 'classEst: ',classEst.T
		#计算下一次迭代中的权重向量D
		expon = multiply(-1*alpha*mat(classLabels,dtype(float)).T,classEst)#计算指数
		D = multiply(D,exp(expon))
		D = D/D.sum()#归一化
		#错误率累加计算
		aggClassEst += alpha*classEst
		#print 'aggClassEst: ',aggClassEst.T
		#aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
		#errorRate = aggErrors.sum()/m
		errorRate = 1.0*sum(sign(aggClassEst)!=mat(classLabels).T)/m#sign(aggClassEst)表示根据aggClassEst的正负号分别标记为1 -1
		print 'total error: ',errorRate
		if errorRate == 0.0:#如果错误率为0那就提前结束for循环
			break
	return weakClassArr


#基于AdaBoost的分类函数
#dataToClass是待分类样例 classifierArr是adaBoostTrainDS函数训练出来的弱分类器数组
def adaClassify(dataToClass,classifierArr):
	dataMatrix = mat(dataToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)): #遍历所有的弱分类器
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
								classifierArr[i]['thread'],\
								classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
	return sign(aggClassEst)




if __name__ == '__main__':
	dataArr,labelArr=loadData('horseColicTraining2.txt')
	classifierArray = adaBoostTrainDS(dataArr,labelArr,10)
	#测试
	testArr, testLabelArr = loadData('horseColicTest2.txt')
	prediction10 = adaClassify(testArr,classifierArray)
	print 1.0*sum(prediction10!=mat(testLabelArr).T)/len(prediction10)

