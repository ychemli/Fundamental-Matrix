// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Modified by Yanis Chemli

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unordered_set>
#include <math.h>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    long int Niter=10000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    float n = matches.size();
    // --------------- >>> MyCode ---------------

    // ---- Normalize matches ----
    float NORM_util[3][3] =     // Normalization Matrix
    {
        {0.001, 0,      0},
        {0,     0.001,  0},
        {0,     0,      1}
    };
    FMatrix<float,3,3> NORM(NORM_util);

    vector<Match> normalized_matches;
    for(int i=0; i<matches.size(); i++) {
        Match m = matches[i];
        m.x1 = m.x1 * 0.001;
        m.y1 = m.y1 * 0.001;
        m.x2 = m.x2 * 0.001;
        m.y2 = m.y2 * 0.001;
        normalized_matches.push_back(m);
    }


    srand ( time(NULL) );       //initalize random seed

    for (long int iter=1; iter<=Niter; iter++){
        // ---- Pick randomly k = 8 points ----

        unordered_set<int> rand_util;       // unique values
        while(rand_util.size()<8){
            int randi = rand() % (int)n;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());

        // ---- Set A matrix ----

        FMatrix<float,9,9> A;
        for (int i=0; i<8;i++){
            Match m = normalized_matches[random_indices[i]];
            float l[] = {m.x1*m.x2, m.y1*m.x2, m.x2, m.x1*m.y2, m.y1*m.y2, m.y2, m.x1, m.y1, 1};
            FVector<float,9> v(l);
            A.setRow(i,v);
        }
        float l9[] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; //add a line of zeros to A to make it a square matrix 9x9
        FVector<float,9> v9(l9);
        A.setRow(8,v9);

        // ---- Singular Value Decomposition(SVD) of A ----

        FVector<float,9> S_A;
        FMatrix<float,9,9> U_A, Vt_A;
        svd(A,U_A,S_A,Vt_A);

        // ---- Retrieve F hat from SVD of A ----

        FVector<float,9> f = Vt_A.getRow(8);
        float F_hat_util[3][3] =                    // 1D to 2D
        {
            {f[0],f[1],f[2]},
            {f[3],f[4],f[5]},
            {f[6],f[7],f[8]}
        };
        FMatrix<float,3,3> F_hat(F_hat_util);

        // ---- Force rank 2 of F hat ----

        FVector<float,3> S_F;
        FMatrix<float,3,3> U_F, Vt_F;
        svd(F_hat,U_F,S_F,Vt_F);                // decompose F hat as SVD

        float S_F_mat_util[3][3] =
        {
            {S_F[0],    0,      0},                   // force s3 = 0
            {0,         S_F[1], 0},
            {0,         0,      0}
        };
        FMatrix<float,3,3> S_F_mat(S_F_mat_util);

        F_hat = U_F * S_F_mat * Vt_F;             // recompose F hat

        // ---- Denormalize F hat => F matrix ----

        FMatrix<float,3,3> F;
        F = transpose(NORM) * F_hat * NORM;

        // ---- Find inliers ----

        vector<int> inliers;

        for(int i=0; i<matches.size(); i++) {
            // x1
            float x1_util[3][1] = {
                {matches[i].x1},
                {matches[i].y1},
                {1}
            };
            FMatrix<float,3,1> X1(x1_util);
            // x2
            float x2_util[3][1] = {
                {matches[i].x2},
                {matches[i].y2},
                {1}
            };
            FMatrix<float,3,1> X2(x2_util);

            FMatrix<float,3,1> right_epi_line(transpose(F)*X1);

            // ---- Compute distance from X2 to right epipolar line FtX1 ----
            float a = right_epi_line(0,0);
            float b = right_epi_line(1,0);
            float c = right_epi_line(2,0);
            float u = X2(0,0);
            float v = X2(1,0);

            float d = abs(a*u+b*v+c)/sqrt(a*a+b*b);

            if(d <= distMax)
                inliers.push_back(i);

        }

        // ---- Update bestInliers and bestF ----

        if(inliers.size() > bestInliers.size()){
            bestInliers.swap(inliers);
            inliers.clear();

            // ---- Recompute number of iterations, Niter ----
            int m = bestInliers.size();

            cout<<"Iter: "<<iter<<"\t No of inliers: "<<m<<endl;

            Niter = ceil( log(BETA) / log(1.0-pow((float)m/n,8.0)) );

            // ---- Refine F, compute SVD from inliners ----

            int r=m;
            Matrix<float> Abest(r,9);
            for (int i=0; i<r;i++){
                Match m = matches[bestInliers[i]];
                float l[] = {m.x1*m.x2, m.y1*m.x2, m.x2, m.x1*m.y2, m.y1*m.y2, m.y2, m.x1, m.y1, 1};
                Vector<float> v(l,9);
                Abest.setRow(i,v);
            }

            // ---- Singular Value Decomposition(SVD) of A ----

            Vector<float> S_Abest(9);
            Matrix<float> U_Abest(r,9), Vt_Abest(9,9);
            svd(Abest,U_Abest,S_Abest,Vt_Abest,true);

            // ---- Retrieve Fbest from SVD of Abest ----

            Vector<float> fbest = Vt_Abest.getRow(8);
            float F_best_util[3][3] =                    // 1D to 2D
            {
                {fbest[0],fbest[1],fbest[2]},
                {fbest[3],fbest[4],fbest[5]},
                {fbest[6],fbest[7],fbest[8]}
            };
            FMatrix<float,3,3> F_best(F_best_util);

            // ---- Force rank 2 of F best ----

            FVector<float,3> S_Fbest;
            FMatrix<float,3,3> U_Fbest, Vt_Fbest;
            svd(F_best,U_Fbest,S_Fbest,Vt_Fbest);                // decompose F best as SVD

            float S_Fbest_mat_util[3][3] =
            {
                {S_Fbest[0],0,0},                   // force s3 = 0
                {0,S_Fbest[1],0},
                {0,0,0}
            };
            FMatrix<float,3,3> S_Fbest_mat(S_Fbest_mat_util);

            bestF = U_Fbest * S_Fbest_mat * Vt_Fbest;             // recompose F best

        }
    }
    // --------------- <<< MyCode ---------------

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++){
        matches.push_back(all[bestInliers[i]]);
    }

    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if(getMouse(x,y) == 3){
        // --------------- TODO ------------
            // x click
            Color color(rand()%256,rand()%256,rand()%256);
            fillCircle(x, y, 2, color);

            float x_click_util[3][1] = {{x},{y},{1}};
            FMatrix<float,3,1> X_click(x_click_util);

            int h = I1.height(); //480
            int w = I1.width();  //640

            int right_flag = 0;     // if right_flag = 1, we have to draw line in the right image

            FMatrix<float,3,1> epi_line;

            if ( x <= w){                           //click in left image
                epi_line = transpose(F)*X_click;    //right epipolar line
                right_flag = 1;
            }
            else{                           //click in right image
                int temp = X_click(0,0);
                X_click(0,0) = temp - w;
                epi_line = F*X_click;       //left epipolar line
                right_flag = 0;
            }

            float a = epi_line(0,0);
            float b = epi_line(1,0);
            float c = epi_line(2,0);

            float borderIntersections [4][2] =      //borders and epipolar line intersection points
            {
                {0, -c/b},          //left
                {w,(-c-a*w)/b},     //right
                {-c/a, 0},          //up
                {(-c-b*h)/a, h}     //down
            };

            vector<float> linePoints;

            for (int i=0; i<4; i++){
                float x = borderIntersections[i][0]+right_flag*w;
                float y = borderIntersections[i][1];
                if( x>=0 && x<=w+right_flag*w && y>=0 && y<=h){
                    linePoints.push_back(x);
                    linePoints.push_back(y);
                }
            }
            if(linePoints.size()>=4)
                drawLine(linePoints[0],linePoints[1],linePoints[2],linePoints[3],color);

         }
    }
}


int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
