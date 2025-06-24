# import PyKDL
# import tf_conversions.posemath
from math import *
import numpy as np

def invTransform(Transform):
	T = np.matrix(Transform)
	R = T[0:3,0:3]
	t = T[0:3,3]

	inverseT = np.hstack((R.transpose(),-R.transpose().dot(t)))
	inverseT = np.vstack((inverseT,[0,0,0,1]))
	return np.asarray(inverseT)

def transformDHParameter(a,d,alpha,theta):
	T = np.array([
			[cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha) ,a*cos(theta)],
			[sin(theta),cos(theta)*cos(alpha) ,-cos(theta)*sin(alpha),a*sin(theta)],
			[0       ,sin(alpha)          ,cos(alpha)          ,d         ],
			[0       ,0                  ,0                  ,1          ]
		])
	return T

def transformRobotParameter(theta):
	d = [0.089159,0,0,0.10915,0.09465,0.0823]
	a = [0,-0.425,-0.39225,0,0,0]
	alpha = [pi/2,0,0,pi/2,-pi/2,0]
	T = np.eye(4)
	for i in range(6):
		T = T.dot(transformDHParameter(a[i],d[i],alpha[i],theta[i]))
	return T

class InverseKinematicsUR5:
	def __init__(self):
		# Debug mode
		self.debug = False

		# Robot DH parameter
		self.d = [0.089159,0,0,0.10915,0.09465,0.0823]
		self.a = [0,-0.425,-0.39225,0,0,0]
		self.alpha = [pi/2,0,0,pi/2,-pi/2,0]

		# Robot EE orientation offset.
		# Useful if the ee orientation when the all joint = 0 is not
		#     1  0  0
		# R = 0  0 -1
		#     0  1  0
		# ee_offset = current_R_all_joint_0.transpose * R
		self.ee_offset = np.eye(4)

		# Robot joint limits
		self.limit_max = 2 * pi
		self.limit_min = -2 * pi

		# Robot joint weights
		self.joint_weights = np.array([1,1,1,1,1,1])

		# Robot target transformation
		self.gd = np.identity(4)

		# Stopping IK calculation flag
		self.stop_flag = False

		# Robot joint solutions data
		self.theta1 = np.zeros(2)

		self.flags1 = None

		self.theta5 = np.zeros((2,2))

		self.flags5 = None

		self.theta6 = np.zeros((2,2))

		self.theta2 = np.zeros((2,2,2))
		self.theta3 = np.zeros((2,2,2))

		self.flags3 = None

		self.theta4 = np.zeros((2,2,2))

	def enableDebugMode(self, debug = True):
		# This function will enable/disable debug mode
		self.debug = debug

	def setJointLimits(self, limit_min, limit_max):
		# This function is used to set the joint limit for all joint
		self.limit_max = limit_max
		self.limit_min = limit_min

	def setJointWeights(self, weights):
		# This function will assign weights list for each joint
		self.joint_weight = np.array(weights)

	def setEERotationOffset(self,r_offset_3x3):
		# This function will assign rotation offset to the ee. r_offset_3x3 should be a numpy array
		self.ee_offset[0:3,0:3] = r_offset_3x3

	def setEERotationOffsetROS(self):
		# This function will assign proper tool orientation offset for ROS ur5's urdf.
		r_offset_3x3 = np.array( [[ 0, 0, 1],[-1, 0, 0],[ 0,-1, 0]] )
		self.setEERotationOffset(r_offset_3x3)

	def normalize(self,value):
		# This function will normalize the joint values according to the joint limit parameters
		normalized = value
		while normalized > self.limit_max:
			normalized -= 2 * pi
		while normalized < self.limit_min:
			normalized += 2* pi
		return normalized

	def getFlags(self,nominator,denominator):
		# This function is used to check whether the joint value will be valid or not
		if denominator == 0:
			return False
		return abs(nominator/denominator) < 1.01

	def getTheta1(self):
		# This function will solve joint 1
		self.flags1 = np.ones(2)

		p05 = self.gd.dot(np.array([0,0,-self.d[5],1]))-np.array([0,0,0,1])
		psi = atan2(p05[1],p05[0])

		L = sqrt(p05[0]**2+p05[1]**2)

		# gives tolerance if acos just a little bit bigger than 1 to return
		# real result, otherwise the solution will be flagged as invalid
		if abs(self.d[3]) > L:
			if self.debug:
				print('L1 = ', L, ' denominator = ', self.d[3])
			self.flags1[:] = self.getFlags(self.d[3],L) # false if the ratio > 1.001
			L = abs(self.d[3])
		phi = acos(self.d[3]/L)

		self.theta1[0] = self.normalize(psi+phi+pi/2)
		self.theta1[1] = self.normalize(psi-phi+pi/2);

		# stop the program early if no solution is possible
		self.stop_flag = not np.any(self.flags1)
		if self.debug:
			print('t1: ', self.theta1)
			print('flags1: ',self.flags1)
	
	def getTheta5(self):
		# This function will solve joint 5
		self.flags5 = np.ones((2,2))

		p06 = self.gd[0:3,3]
		for i in range(2):
			p16z = p06[0]*sin(self.theta1[i])-p06[1]*cos(self.theta1[i]);
			L = self.d[5]

			if abs(p16z - self.d[3]) > L:
				if self.debug:
					print('L5 = ', L, ' denominator = ', abs(p16z - self.d[3]))
				self.flags5[i,:] = self.getFlags(p16z - self.d[3],self.d[5])
				L = abs(p16z-self.d[3]);
			theta5i = acos((p16z-self.d[3])/L)
			self.theta5[i,0] = theta5i
			self.theta5[i,1] = -theta5i

		# stop the program early if no solution is possible
		self.stop_flag = not np.any(self.flags5)
		if self.debug:
			print('t5: ', self.theta5)
			print('flags5: ',self.flags5)

	def getTheta6(self):
		# This function will solve joint 6
		for i in range(2):
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
			T61 = invTransform(invTransform(T1).dot(self.gd))
			for j in range(2):
				if sin(self.theta5[i,j]) == 0:
					if self.debug:
						print("Singular case. selected theta 6 = 0")
					self.theta6[i,j] = 0
				else:
					self.theta6[i,j] = atan2(-T61[1,2]/sin(self.theta5[i,j]),
											  T61[0,2]/sin(self.theta5[i,j]))
		# print 't6: ', self.theta6

	def getTheta23(self):
		# This function will solve joint 2 and 3
		self.flags3 = np.ones ((2,2,2))
		for i in range(2):
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
			T16 = invTransform(T1).dot(self.gd)
			
			for j in range(2):
				T45 = transformDHParameter(self.a[4],self.d[4],self.alpha[4],self.theta5[i,j])
				T56 = transformDHParameter(self.a[5],self.d[5],self.alpha[5],self.theta6[i,j])
				T14 = T16.dot(invTransform(T45.dot(T56)))

				P13 = T14.dot(np.array([0,-self.d[3],0,1]))-np.array([0,0,0,1])
				L = P13.dot(P13.transpose()) - self.a[1]**2 - self.a[2]**2

				if abs(L / (2*self.a[1]*self.a[2]) ) > 1:
					if self.debug:
						print('L3 = ', L, ' denominator = ', (2*self.a[1]*self.a[2]))
					self.flags3[i,j,:] = self.getFlags(L,2*self.a[1]*self.a[2])
					L = np.sign(L) * 2*self.a[1]*self.a[2]
				self.theta3[i,j,0] = acos(L / (2*self.a[1]*self.a[2]) )
				self.theta2[i,j,0] = -atan2(P13[1],-P13[0]) + asin( self.a[2]*sin(self.theta3[i,j,0])/np.linalg.norm(P13) )
				self.theta3[i,j,1] = -self.theta3[i,j,0]
				self.theta2[i,j,1] = -atan2(P13[1],-P13[0]) + asin( self.a[2]*sin(self.theta3[i,j,1])/np.linalg.norm(P13) )
		if self.debug:
			print('t2: ', self.theta2)
			print('t3: ', self.theta3)
			print('flags3: ',self.flags3)

		# stop the program early if no solution is possible
		self.stop_flag = not np.any(self.flags3)
	
	def getTheta4(self):
		# This function will solve joint 4 value
		for i in range(2):
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
			T16 = invTransform(T1).dot(self.gd)
			
			for j in range(2):
				T45 = transformDHParameter(self.a[4],self.d[4],self.alpha[4],self.theta5[i,j])
				T56 = transformDHParameter(self.a[5],self.d[5],self.alpha[5],self.theta6[i,j])
				T14 = T16.dot(invTransform(T45.dot(T56)))

				for k in range(2):
					T13 = transformDHParameter(self.a[1],self.d[1],self.alpha[1],self.theta2[i,j,k]).dot(
						  transformDHParameter(self.a[2],self.d[2],self.alpha[2],self.theta3[i,j,k]) )
					T34 = invTransform(T13).dot(T14)
					self.theta4[i,j,k] = atan2(T34[1,0],T34[0,0])
		if self.debug:
			print('t4: ', self.theta4)

	def countValidSolution(self):
		# This function will count the number of available valid solutions
		number_of_solution = 0
		for i in range(2):
			for j in range(2):
				for k in range(2):
					if self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]:
						number_of_solution += 1
		return number_of_solution

	def getSolution(self):
		# This function will call all function to get all of the joint solutions
		for i in range(4):
			if i == 0:
				self.getTheta1()
			elif i == 1:
				self.getTheta5()
			elif i == 2:
				self.getTheta6()
				self.getTheta23()
			elif i == 3:
				self.getTheta4()

			# This will stop the solving the IK when there is no valid solution from previous joint calculation
			if self.stop_flag:
				return

	def solveIK(self,forward_kinematics):
		self.gd = forward_kinematics.dot(self.ee_offset)
		if self.debug:
			print('Input to IK:\n', self.gd)
		self.getSolution()
		number_of_solution = self.countValidSolution()
		if self.stop_flag or number_of_solution < 1:
			if self.debug:
				print('No solution')
			return None

		Q = np.zeros((number_of_solution,6))
		index = 0
		for i in range(2):
			for j in range(2):
				for k in range(2):
					if not (self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]):
						# skip invalid solution
						continue
					Q[index,0] = self.normalize(self.theta1[i])
					Q[index,1] = self.normalize(self.theta2[i,j,k])
					Q[index,2] = self.normalize(self.theta3[i,j,k])
					Q[index,3] = self.normalize(self.theta4[i,j,k])
					Q[index,4] = self.normalize(self.theta5[i,j])
					Q[index,5] = self.normalize(self.theta6[i,j])
					index += 1

		if self.debug:
			print('Number of solution: ', number_of_solution)
			print(Q)

		return Q

	def findClosestIK(self,forward_kinematics,current_joint_configuration):
		current_joint = np.array(current_joint_configuration)
		Q = self.solveIK(forward_kinematics)
		if Q is not None:
			delta_Q = np.absolute(Q - current_joint) * self.joint_weights
			delta_Q_weights = np.sum(delta_Q, axis=1)
			closest_ik_index = np.argmin(delta_Q_weights, axis = 0)

			if self.debug:
				print('delta_Q weights for each solutions:', delta_Q_weights)
				print('Closest IK solution: ', Q[closest_ik_index,:])
			return Q[closest_ik_index,:]
		else:
			# print("no solution found")
			return None
