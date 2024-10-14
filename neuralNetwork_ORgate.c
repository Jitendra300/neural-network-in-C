#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define exp 2.7182

// defining global variables
int trainingIteration = 10000;
/* this character stores which activatioin is used
s for sigmoid function
t for tanh function
r for ReLu function
 */
char activationUsed = 's';

// defining our Matrices here....
double trainingInputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double trainingOuputs[1][4] = {{0, 1, 1, 1}}; // Truth Table Value for OR gate....
double trainingInputsTranspose[2][4] = {{0, 0, 1, 1}, {0, 1, 0, 1}};

// Input neurons weights
double synaptic_weights[2];
// here for sake of simplicity we have taken Output neurons as 4 instead of 2, for two i guess we would have to have some internal layer and then keep 2 output neuron
double outputArr[4];
// This matrice will keep how off it is from real prediction
double errorArr[4];
// This matrice will keep changing weights based on how off the neural network is from real prediction
double adjustmentArr[2];

double activationFunc(double x, char whichFunc){
    if(whichFunc == 's') return 1.0 / (1+pow(exp, -x)); // this is sigmoid function
    /* if(whichFunc == 't') return (pow(exp, 2*x) - 1.0) / (pow(exp, 2*x) + 1.0); // this is tanh function */
    return 0;
}

double activationFuncDerivative(double x, char whichFunc){
    if(whichFunc == 's') return x*(1-x);
    /* if(whichFunc == 't') return  */
    return 0;
}

// this  function does some activation function operation or activation function derivative operation;
// d for derivative  operation && s for function operation
void activation_on_matrix(int r1, int c1, double outputArr[][c1], char operation){
    for(int i=0;i<r1;++i){
        for(int j=0;j<c1;++j){
            if(operation == 's')
                outputArr[i][j] = activationFunc(outputArr[i][j], activationUsed);
            else
                outputArr[i][j] = activationFuncDerivative(outputArr[i][j], activationUsed);
        }
    }
}

// here we can add, subtract, or do normal multiplication[lets consider the matrice as scalar and do it....ik its not called multiplication but fine....]two matrices...
void addOrSubOrMulMatrices(int r, int c, double A[][c], double B[][c], double C[][c], char operation){
    for(int i=0;i<r;++i){
        for(int j=0;j<c;++j){
            if(operation == '+') C[i][j] = A[i][j] + B[i][j];
            else if(operation == '-') C[i][j] = A[i][j] - B[i][j];
            else C[i][j] = A[i][j] * B[i][j];
        }
    }
}

void dotProduct(int r1, int c1, int r2, int  c2, double A[][c1], double B[][c2], double output[][c2]){
    for(int i=0;i<r1;++i){
        for(int j=0;j<c2;++j) output[i][j] = 0;
    }

    for(int i=0;i<r1;++i){
        for(int j=0;j<c2;++j){
            for(int k=0;k<c1;++k){
                output[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void train(double trainingIn[][2], double trainingOp[], int total_iteration){
    for(int current_iteration=0;current_iteration<total_iteration;++current_iteration){
        // Getting output based on our current weights 
        dotProduct(4, 2, 2, 1, trainingInputs, synaptic_weights, outputArr);
        activation_on_matrix(4, 1, outputArr, 's'); // this is doing function operation on matrice
        //calculating error...
        addOrSubOrMulMatrices(4, 1, trainingOp, outputArr, errorArr, '-');
        activation_on_matrix(4, 1, outputArr, 'd'); // this is doing derivation of function operation on matrice
        addOrSubOrMulMatrices(4, 1, outputArr, errorArr, outputArr, '*');
        // Getting some adjustment matrices
        dotProduct(2, 4, 4, 1, trainingInputsTranspose, outputArr, adjustmentArr);
        // changing our weights as per adjustment
        addOrSubOrMulMatrices(2, 1, synaptic_weights, adjustmentArr, synaptic_weights, '+');
    }
}

void initialization(void){
    srand(time(NULL)); // this is adding a seed for random generation....not needed but fine...
    printf("\nStarting Random Weights: \n");
    for(int i=0;i<2;++i){
        double value = (double)rand() / RAND_MAX * 2.0 - 1;
        synaptic_weights[i] = value;
        printf("%lf ", value);
    }
    printf("\n\n");
}

int main(void){
    initialization();
    
    train(trainingInputs, trainingOuputs, 10000);
    
    printf("Synaptic Weights After Training: \n");
    for(int i=0;i<2;++i) printf("%f ", synaptic_weights[i]);
    
    printf("\n\n");
    
    double input1, input2;
    printf("Input 1: "); scanf("%lf", &input1);
    printf("Input 2: "); scanf("%lf", &input2);

    printf("\nSituation: %.0lf %.0lf \n\n", input1, input2);

    double arr[1][2] = {{input1, input2}};

    dotProduct(1, 2, 2, 1, arr, synaptic_weights, outputArr);
    double prediction = activationFunc(outputArr[0], 's');
    printf("Prediction: %lf \n\n", prediction);    

    printf("So...output according to Neural Network is:  ");
    if(prediction >= 0.90) printf("1");
    else printf("0");
    
    return 0;
}
