#
# n-armed bandit task chapter 2 Figure 2.5
#
# 1. implemented with softmax function or epsilon-greedy function
# 2. implemented with online method for values
# 3. implemented with comparison test
#

import numpy as np
import matplotlib.pyplot as plt

#nBandit: the number of bandits
#nArm: the number of arms
#nPlay: the number of plyas (times we will pull a arm)
#sigma the standard deviation of the return from each of the arms

#Epsilon Greedy Method: x=qt, y=teps
def Greedy_method(x, y, z):
	qt = x[0];
	if np.random.rand() <= y: #epsilon part
		arm = np.random.randint(0, z);
	else: #greedy part
		#arm = np.where(max(qT[bi, :])==qT[bi,:])[0][0];
		arm = np.argmax(qt);
	return arm

#SAMPLE_DISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
#M = sample_discrete(prob, r, c)
#Example: sample_discrete([0.8 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
#where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.
def sample_discrete(prob, r=1, c=1):
	n = len(prob[0]);
	R = np.random.rand(r, c);
	M = np.ones((r, c));
	cumprob = np.cumsum(prob[:]);
	if n < r*c:
		for i in xrange(n):
			M = M + (R > cumprob[i]);
	else:
		cumprob2 = cumprob[0:-1];
		for i in xrange(r):
			for j in xrange(c):
				M[i, j] = np.sum(R[i,j] > cumprob2);
	return M

def comparison_test(nBandit=2000, nArm=10, nPlay=1000, sigma=1.0, update=1):
	#epsilon
	tEPs = 0.1;
	#alpha: learning rate
	alpha = 0.1;
	#beta
	beta = 0.1;

	Average_Reward = np.zeros((3, nPlay));
	perOptAction = np.zeros((3, nPlay));

	OptAction = np.zeros((3, nBandit, nPlay))
	AllRewards = np.zeros((3, nBandit, nPlay));

	for bi in xrange(nBandit):
		mu = 0; sigma = 1;
		qTmean = np.random.normal(mu, sigma, nArm);
		qT_M1 = np.zeros((1, nArm));
		qT_M2 = np.zeros((1, nArm));
		qN_M2 = np.zeros((1, nArm));

		#reinforcement comparison method:
		pT = np.zeros((1, nArm)); #initialize play preference
		rT = 0.0; #initialize a reference reward

		#make loop
		for p in xrange(nPlay):
			arm_M1 = Greedy_method(qT_M1, tEPs, nArm);
			arm_M2 = Greedy_method(qT_M2, tEPs, nArm);

			e = np.exp(pT);
			piT = e/np.sum(e);
			#draw an arm from the distribution piT
			arm_M3 = sample_discrete(piT, 1, 1)[0][0];

			best_arm = np.argmax(qTmean[:]);
			if arm_M1 == best_arm: OptAction[0, bi, p] = 1.0;
			if arm_M2 == best_arm: OptAction[1, bi, p] = 1.0;
			if arm_M3 == best_arm: OptAction[2, bi, p] = 1.0;

			#reward
			reward_M1 = qTmean[arm_M1] + sigma * np.random.normal();
			reward_M2 = qTmean[arm_M2] + sigma * np.random.normal();
			reward_M3 = qTmean[arm_M3] + sigma * np.random.normal();

			AllRewards[0, bi, p] = reward_M1;
			AllRewards[1, bi, p] = reward_M2;
			AllRewards[2, bi, p] = reward_M3;

			#update qT
			qT_M1[0, arm_M1] = qT_M1[0, arm_M1] + alpha * (reward_M1 - qT_M1[0, arm_M1]);
			qT_M2[0, arm_M2] = qT_M2[0, arm_M2] + (reward_M2 - qT_M2[0, arm_M2])/(qN_M2[0, arm_M2] + 1.0);
			qN_M2[0, arm_M2] = qN_M2[0, arm_M2] + 1.0;
			
			#The reinforcement comparison update
			if update == 0:
				pT[0, arm_M3] = pT[0, arm_M3] + beta * (reward_M3 - rT);
			else:
				pT[0, arm_M3] = pT[0, arm_M3] + beta * (reward_M3 - rT) * (1.0 - piT[0, arm_M3]);

			rT = rT + alpha * (reward_M3 - rT);
	for i in xrange(3):
		avg = np.mean(AllRewards[i], 0);
		print avg
		Average_Reward[i] = avg.T;
		Opt = np.mean(OptAction[i], 0);
		perOptAction[i] = Opt.T;

	#Plotting
	x = np.linspace(0, nPlay, nPlay);
	for i in xrange(3):
		plt.plot(x, Average_Reward[i]);

	plt.xlabel("nPlays");
	plt.ylabel("Rewards");
	plt.title("reinforcement comparison test");
	plt.legend(["alpha = 0.1", "alpha = 1/k", "rein\_comp"]);
	plt.show()

	for i in xrange(3):
		plt.plot(x, perOptAction[i]*100);

	plt.xlabel("nPlays");
	plt.ylabel("Optimal Action(%)");
	plt.title("reinforcement comparison test");
	plt.legend(["alpha = 0.1", "alpha = 1/k", "rein\_comp"]);
	plt.show()

def main():
	comparison_test();


if __name__ == "__main__":
	main()







