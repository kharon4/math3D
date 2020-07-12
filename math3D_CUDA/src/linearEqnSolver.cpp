#include "linearEqnSolver.h"


namespace LES {

	//equation :
	//eqn::eqn(){ constant = 0; }


	eqn::eqn(unsigned long int noCoefficients, double* coeff, double Constant) {
		constant = Constant;
		coefficients.resize(noCoefficients);

		if (coeff != nullptr)
			for (unsigned long int i = 0; i < noCoefficients; ++i) {
				coefficients[i] = coeff[i];
			}
		else
			for (unsigned long int i = 0; i < noCoefficients; ++i) {
				coefficients[i] = 0;
			}
	}

	inline unsigned long int eqn::noCoeff() const { return coefficients.size(); }

	inline void eqn::setNoCoeff(unsigned long int no) { 
		coefficients.resize(no); 
		for (unsigned long int i = 0; i < no; ++i)coefficients[i] = 0; 
	}

	double eqn::getCoeff(unsigned long int index) const {
		if (index < coefficients.size())return coefficients[index];
		else if (index == coefficients.size())return constant;
		return 0;
	}

	bool eqn::setCoeff(unsigned long int index, double val) {
		if (index < coefficients.size())coefficients[index] = val;
		else if (index == coefficients.size())constant = val;
		else return true;

		return false;
	}

	inline double eqn::getConst() const { return constant; }

	inline void eqn::setConst(double val) { constant = val; }

	void eqn::displayEqn(std::ostream& f, bool showVar) {
		if (!showVar) {
			if (coefficients.size() > 0)f << coefficients[0];
			for (unsigned long int i = 1; i < coefficients.size(); ++i) {
				f << " , " << coefficients[i];
			}
			f << " | " << constant;
		}
		else {
			if (coefficients.size() > 0)f << coefficients[0] << "*x0";
			for (unsigned long int i = 1; i < coefficients.size(); ++i) {
				f << " + " << coefficients[i] << "*x" << i;
			}
			f << " = " << constant;
		}

	}


	///System functions
	void system::calculateSysType() {
		if (noEqns < noCoef) {
			systemType = sysType::underDefined;
		}
		else if (noEqns > noCoef) {
			systemType = sysType::overDefined;
		}
		else {
			systemType = sysType::criticallyDefined;
		}
	}


	system::system(unsigned long int noCoefficients, unsigned long int noEquations) {
		solved = false;
		noEqns = noEquations;
		noCoef = noCoefficients;
		eqns = new eqn[noEqns];
		for (unsigned long int i = 0; i < noEqns; ++i) {
			eqns[i].setNoCoeff(noCoefficients);
		}
		
		calculateSysType();
	}

	inline unsigned long int system::getNoCoeff() const { return noCoef; }
	inline unsigned long int system::getNoEqns() const { return noEqns; }
	inline system::sysType system::getSystemType()const { return systemType; }


	inline void system::changeNoCoeff(unsigned long int noC) {
		solved = false;
		noCoef = noC;
		for (unsigned long int i = 0; i < noEqns; ++i)eqns[i].setNoCoeff(noC);
		calculateSysType();
	}

	inline void system::changeNoEqns(unsigned long int noE) {
		solved = false;
		delete[] eqns;
		eqns = nullptr;
		eqns = new eqn[noE];
		noEqns = noE;
		for (unsigned long int i = 0; i < noE; ++i)eqns[i].setNoCoeff(noCoef);
		calculateSysType();
	}

	inline void system::changeSystemSize(unsigned long int noC, unsigned long int noE) {
		solved = false;
		noCoef = noC;
		noEqns = noE;
		delete[] eqns;
		eqns = nullptr;
		eqns = new eqn[noE];
		for (unsigned long int i = 0; i < noE; ++i)eqns[i].setNoCoeff(noC);
		calculateSysType();
	}

	void system::displayMatrix(std::ostream& f, bool showVar) const {
		for (unsigned long int i = 0; i < noEqns; ++i) {
			eqns[i].displayEqn(f, showVar);
			f << std::endl;
		}
	}

	void system::load(std::istream& f) {
		solved = false;
		unsigned long  int noC, noE;
		f >> noC >> noE;
		changeSystemSize(noC, noE);
		for (unsigned long int i = 0; i < noE; ++i) {
			for (unsigned long int j = 0; j <= noC; ++j) {
				double temp;
				f >> temp;
				eqns[i].setCoeff(j, temp);
			}
		}
	}

	void system::load(unsigned long int noC, unsigned long int noE, double* arr) {
		solved = false;
		changeSystemSize(noC, noE);
		for (unsigned long int i = 0; i < noE; ++i) {
			for (unsigned long int j = 0; j <= noC; ++j) {
				eqns[i].setCoeff(j, arr[i* (noC+1) + j]);
			}
		}
	}

	void system::solve() {
		if (solved)return;
		unsigned long int currentCno = 0;
		unsigned long int currentEqn = 0;
		unsigned long int minItt = ((noCoef < noEqns) ? noCoef : noEqns);//find minimum itterations
		for (unsigned long int i = 0; i < minItt; ++i) {//calculate REF
			//find a eqn with non 0 coeff at currentCno;
			{
				unsigned long int j;
				for (j = currentEqn; j < noEqns; ++j) {
					if (eqns[j].getCoeff(currentCno) != 0) {//replace the eqn and get it to the top
						eqn temp = eqns[currentEqn];
						eqns[currentEqn] = eqns[j];
						eqns[j] = temp;
						break;
					}
				}
				if (j == noEqns) {//increment cno and goto the next itteration
					currentCno++;
					continue;
				}
			}
			//perform row opperations on the matrix
			for (unsigned long int j = currentEqn + 1; j < noEqns; ++j) {
				//calculate the coefficient
				double coeff = -(eqns[j].getCoeff(currentCno)) / eqns[currentEqn].getCoeff(currentCno);
				eqns[j].setCoeff(currentCno, 0);
				//calculate the rest of the coefficients
				for (unsigned long int k = currentCno + 1; k <= noCoef; ++k) {
					eqns[j].setCoeff(k, eqns[j].getCoeff(k) + coeff * eqns[currentEqn].getCoeff(k));
				}
			}
			currentCno++;
			currentEqn++;
		}

		//predict outcome
		//calculate no of 0 rows;
		solutionType = solType::uniqueSol;
		unsigned long int zRows = 0;
		for (long long int i = noEqns - 1; i >= 0; i--) {
			bool allZ = true;
			for (long long int j = noCoef - 1; j >= 0; j--) {
				if (eqns[i].getCoeff(j) != 0) {
					allZ = false;
					break;
				}
			}
			if (allZ) {
				zRows++;
				if (eqns[i].getConst() != 0) {
					solutionType = solType::inconsistant;
					break;
				}
			}
			else {
				break;
			}
		}
		//calculate the thing
		if (solutionType != solType::inconsistant) {
			if (noCoef <= (noEqns - zRows)) {
				solutionType = solType::uniqueSol;
			}
			else {
				solutionType = solType::infiniteSols;
			}
		}

		//calculate the solution if unique
		if (solutionType == solType::uniqueSol) {
			sols.resize(noCoef);
			for (long long int i = noEqns - zRows - 1; i >= 0; i--) {//for every row
				//calculate sum;
				double sum = 0;
				for (unsigned long int j = noCoef - 1; j > i; j--) {
					sum += sols[j] * eqns[i].getCoeff(j);
				}
				sols[i] = (eqns[i].getConst() - sum) / eqns[i].getCoeff(i);
			}
		}
		else {
			sols.resize(0);
		}
		solved = true;
	}

	system::solType system::getSolution(std::vector<double>& solutions) {
		if (!solved) solve();
		solutions.resize(0);
		for (unsigned long int i = 0; i < sols.size(); ++i)solutions.push_back(sols[i]);
		return solutionType;
	}

	system::~system() { delete[] eqns; }

}