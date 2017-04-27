#include <stdio.h>
#include <math.h>
#include "jpg.h"
#include "mnist.h"
#include "limits.h"

float dist(float* v1, float* v2) {
  float d=0;
  for (int i=0; i<784; i++) {
    d += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return d;
}

int lin_class(float* w, float* x) {
	float d=0;
	for(int i=0;i<784;i++) {
		d+= w[i]*x[i];
	}
	if(d>=0) return 1;
	else return -1;
}

	const int K = 10;
	float A[K][784];
	float B[K][784];



int main(int argc, char** argv) {
	float** images = read_mnist("train-images.idx3-ubyte");
	float* labels = read_labels("train-labels.idx1-ubyte");
	float** test_images = read_mnist("t10k-images.idx3-ubyte");
	float* test_labels = read_labels("t10k-labels.idx1-ubyte");
	float* w = new float[784];

	//Partie K means
	
	int* n = new int[K];
	
	for(int i=0; i<K ; i++) {
		for(int j=0; j<784; j++) {
			A[i][j] = (float)rand()*2/INT_MAX-1;
			B[i][j] = 0;
		}
	}
	
	//2) Main loop
	for(int t=0; t<1000;t++) {
	
		//Step 1
		for(int i=0; i<K ; i++) {
			n[i] = 0;
			for(int j=0; j<784; j++) {
				B[i][j]=0;
			}
		}
	
		//Step 2
		for(int i=0; i<60000;i++) {
			float mind = -1; int gagnant = 0;
			for(int k=0; k<K; k++) {
				float d = dist(A[k],images[i]);
				if(d <= mind || mind == -1) {
					mind = d,gagnant= k;
				}
			}

			//Accumulation des images
			for(int j=0;j<784;j++) {
				B[gagnant][j] += images[i][j];
			}
			n[gagnant]++;
		}
	
		//Step 3	
		for(int k=0; k<K; k++) {
			for(int j=0; j<784; j++) {
				A[k][j] = B[k][j]/n[k];
			}
		}
	
		//Step 4 Sauvegarde
		for(int k=0; k<K; k++) {
			save_jpg(A[k], 28, 28, "%u/%u.jpg", k, t);
		}
	}




	//STEP 1 : INITIALISATION 
	for (int i=0; i<784; i++) w[i]=(float)rand()*2/INT_MAX;
	float gamma = 0.01;


	//STEp 2 : LEARNING (que les données de train)
	for (int i=0; i<100; i++) {
			//Calcul gradient ( g= y*x si erreur 0, sinon )
		int prediction = lin_class(w,images[i]);
		int verite = (labels[i] == 1) ? 1 : -1;
		if (verite !=prediction) {
			printf("ERREUR\n");
			//(w[t+1] = w[t] - gamma*y*x)
			for(int j=0; j<784; j++) {
				w[j] = w[j] + gamma * verite * images[i][j];
			}
		}
	}

	//STEP 3 : TEST (que les données de test)
	float E = 0;
	for(int i=0; i<10000; i++) {
		printf("%u\n",i);
    	int inf = lin_class(w,test_images[i]);

    	//save_jpg(test_images[i], 28, 28, "%u/%u.jpg", inf,i);

	if ((inf==1 && test_labels[i]!=1)||(inf== -1 && test_labels[i]==1)) E++;

		printf("Erreur = %0.2f%%\n",(E*100)/i);
    }

    return 0;
}
