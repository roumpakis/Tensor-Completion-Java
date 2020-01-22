package Utilities;


import Jama.QRDecomposition;


import java.util.ArrayList;
import java.util.Random;
import la.matrix.DenseMatrix;
import la.vector.DenseVector;
import tensorcompletion.Tensor;


import java.nio.file.*;

  

public class TMacPar {

/* 

     */
    double relerr1,relerr2;

    int nstall;
    final static double EPSILON = 0.0000001;
    private int maxT; // maximum Time
    private int maxit = 10;// maximum Iteerations
    private  double tol = 1e-8;
    private int N = 3;

    Matlab matlab = new Matlab();
   // increase scheme strategy

    double alpha_adj = 0;

    DenseMatrix rank_adj ;
    DenseMatrix rank_inc ;
    DenseMatrix rank_min ;
    DenseMatrix rank_max ;
    DenseMatrix alpha;


    DenseMatrix Nway;
    DenseMatrix coreNway ;
    DenseMatrix coNway;

//    ArrayList<DenseMatrix> X ;
//    ArrayList<DenseMatrix> Y ;


    ArrayList<DenseMatrix> sx;
    DenseMatrix reschg ;
    double reschg_tol;


    DenseMatrix res0;
    DenseMatrix res ;
    double TotalRes ;


    Tensor M ;
    ArrayList<Tensor> Mn ;
    int rank_inc_num;


    ArrayList<DenseMatrix> Xsq;
    ArrayList<DenseMatrix> Yt ;
    ArrayList<DenseMatrix> spI;
    DenseMatrix solX ;


    ArrayList<DenseMatrix> X0 ;
    ArrayList<DenseMatrix> Y0 ;
    Tensor M0;


    double[][] MnUnfolding;
    DenseMatrix MnUnfoldingDense ;

    double tic;

    
            final String dir = System.getProperty("user.dir");
        String path = dir+"\\data\\";
        String pathRes = dir+"\\res\\";
        
    public ArrayList<Tensor> TMAC(Tensor tensor,ArrayList<DenseMatrix> X,ArrayList<DenseMatrix> Y, ArrayList<TensorIndex> known,ArrayList<Double> data, boolean isTest ,int W, double overlap){

        double[][][] tensor1 = tensor.getTensor();
        int XSize = tensor1.length;
        int YSize = tensor1[0].length;
        int ZSize = tensor1[0][0].length;
        
System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TMac Par:  >>>>>>>>.");
        TensorIndex mapLinearTI = tensor.mapLinearTI(XSize*YSize*ZSize);

// alpha where sum(alpha) == 1 ---- {weights}
         alpha = new DenseMatrix(1, N);
        for (int i = 0; i < N; i++) {
            // 1/N=3 = 0.333333333
            alpha.setEntry(0, i, 0.333333333);
        }


        // Rank vectors default values
         rank_adj = (DenseMatrix) ml.utils.Matlab.ones(1, N);
         rank_inc = (DenseMatrix) ml.utils.Matlab.ones(1, N);
         rank_min = (DenseMatrix) ml.utils.Matlab.ones(1, N);
         rank_max = (DenseMatrix) ml.utils.Matlab.ones(1, N);

        for(int i = 0; i<N; i++) rank_min.setEntry(0,i,1);

//data: observed entries of the underlying tensor
      //  ArrayList<Double> data = tensor.TensorObservedElements();


//    known: indices of observed entries
      //  ArrayList<TensorIndex> known = tensor.TensorObservedIndecies();


        int I = tensor.getTensor().length;
        int J = tensor.getTensor()[0].length;
        int K = tensor.getTensor()[0][0].length;


        // oi Pragmatikes diastaseis tou Tensor
         Nway = new DenseMatrix(1, 3);
        Nway.setEntry(0, 0, I);
        Nway.setEntry(0, 1, J);
        Nway.setEntry(0, 2, K);

// estimated ranks of all mode matricizations defaults ranks = 10
         coreNway = new DenseMatrix(1, 3);
        coreNway.setEntry(0, 0, 0.15 * Nway.getEntry(0, 0));
        coreNway.setEntry(0, 1, 0.15 * Nway.getEntry(0, 1));
        coreNway.setEntry(0, 2,0.15 * Nway.getEntry(0, 2));
/* coreNway =  estimatedRank1.....*/

        //  the co Nway
         coNway = new DenseMatrix(1, N);
        for (int i = 0; i < N; i++) {
            double integer = matlab.prod(Nway) / Nway.getEntry(0, i);
            coNway.setEntry(0, i, integer);
        }



        for (int i = 0; i < N; i++) {
            int rr= (int) (Nway.getEntry(0,i));
            rank_max.setEntry(0, i,1.5 * rr);
        }



        // !!!!!!!!!!!!!!!!!!!!!!!  rescale the initial point based on number of elements !!!!!!!!!!!!!!!!!!!!!!!
        DenseVector listtoDenseVector = matlab.ArrayListtoDenseVector(data);
        Double nrmb = ml.utils.Matlab.norm(listtoDenseVector);

        double estMnrm = Math.sqrt( (nrmb*nrmb) *(matlab.prod(Nway)/known.size()));



        for (int i = 0; i < N; i++) {
            Double FrobXn = ml.utils.Matlab.norm(X.get(i), "fro");
            Double FrobYn = ml.utils.Matlab.norm(Y.get(i), "fro");

            X.set(i, (DenseMatrix) ml.utils.Matlab.times(X.get(i),1/FrobXn));                                                  //   X{n} = X{n}/norm(X{n},'fro')*estMnrm^(Nway(n)/(Nway(n)+coNway(n)));
            Y.set(i,(DenseMatrix) ml.utils.Matlab.times(Y.get(i),1/FrobYn));

            int under = (int) (Nway.getEntry(0,i) + coNway.getEntry(0,i));

            float powX = (float)Nway.getEntry(0,i) / under ;
            float powY = (float)coNway.getEntry(0,i) / under;

            double XnScale = Math.pow(estMnrm,powX );
            double YnScale =  Math.pow(estMnrm,powY) ;


            X.set(i, (DenseMatrix) X.get(i).times(XnScale));                                                  //   X{n} = X{n}/norm(X{n},'fro')*estMnrm^(Nway(n)/(Nway(n)+coNway(n)));
            Y.set(i, (DenseMatrix) Y.get(i).times(YnScale));                                                  //   Y{n} = Y{n}/norm(Y{n},'fro')*estMnrm^(coNway(n)/(Nway(n)+coNway(n)));
        }





        //Copy X & Y
        X0 = new ArrayList<>();
        Y0 = new ArrayList<DenseMatrix>();
        for (int i = 0; i < N; i++) {
            X0.add(i, matlab.DenseMatrixCopy(X.get(i)));
            Y0.add(i, matlab.DenseMatrixCopy(Y.get(i)));
        }
/* -------------------------------------------
[known,id] = sort(known);
    data = data(id);
 ---------------------------------------------*/

        sx = new ArrayList<>();                                                                       //sx = cell(1,N);
        reschg = (DenseMatrix) ml.utils.Matlab.ones(1, N);                                            //reschg = ones(1,N);
         reschg_tol = Math.max(1e-2, 10 * tol);                                                        //reschg_tol = max(1e-2,10*tol);
        rank_inc_num = matlab.sumOne(rank_adj);


         res0 = (DenseMatrix) ml.utils.Matlab.zeros(1, N);                                              //res0 = zeros(1,N);
         res = matlab.DenseMatrixCopy(res0);                                                            //res = res0;
         TotalRes = 0.0;                                                                                //TotalRes = 0;

        int timestamps = (int) Nway.getEntry(0, 0);
        int devices = (int) Nway.getEntry(0, 1);
        int modalities = (int) Nway.getEntry(0, 2);

        // M = zeros(Nway);
         M = new Tensor(timestamps,devices,modalities);                                               // M = zeros(Nway);

        //M(known) = data;
      M.copyTensorObserved(data, known);                                                               //M(known) = data;



         Mn = new ArrayList<>();
        DenseVector NwayDV = new DenseVector(3);
        NwayDV.set(0,Nway.getEntry(0,0));
        NwayDV.set(1,Nway.getEntry(0,1));
        NwayDV.set(2,Nway.getEntry(0,2));
        for (int i = 0; i < N; i++) {                                                                       //  for n = 1:N {
//
            DenseMatrix XtimesY = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(i), Y.get(i));
            Tensor folding = matlab.Fold(XtimesY,NwayDV,i);
            Mn.add(i, folding);                                                                          //   Mn = Fold(X{n}*Y{n},Nway,n);
            ArrayList<Double> estimationKnownDiff = matlab.EstimationKnownDiff(Mn.get(i), data, known);
            DenseVector estimationKnownDense = matlab.ArrayListtoDenseVector(estimationKnownDiff);
            res0.setEntry(0, i, ml.utils.Matlab.norm(estimationKnownDense));           //res0(n) = norm(Mn(known)-data);

            TotalRes = TotalRes + res0.getEntry(0, i);                                                     // TotalRes = TotalRes+res0(n);
//
        }
        solX = (DenseMatrix) ml.utils.Matlab.ones(1, N); //solX = ones(1,N);
        Xsq = new ArrayList<>();                                                                       //Xsq = cell(1,N);
        Xsq.add(0,new DenseMatrix());
        Xsq.add(1,new DenseMatrix());
        Xsq.add(2,new DenseMatrix());

        Yt = new ArrayList<>();                                                                         // Yt = cell(1,N);
       spI = new ArrayList<>();                                                                         //spI = cell(1,N);


        for (int i = 0; i < N; i++) { //for n = 1:N
           // System.out.println("tou psiiiiiiiiiiii Yt1");
            Yt.add(i, (DenseMatrix) Y.get(i).transpose());                                               // Yt{n} = Y{n}';
        }                                                                                                //end

        //*********************************** start_time = tic;
        tic = ml.utils.Time.tic();



        double SumAlpha = matlab.sum(alpha);
        for(int i = 0; i<N; i++){
            alpha.setEntry(0,i,(matlab.DenseMatrixConstDiv(alpha,SumAlpha)).getEntry(0,i));             // alpha/sum(alpha);
        }


        for (int k = 0; k < maxit; k++) {
            System.out.println("i = "+k+":"+maxit);
//UpdateX&Y

            for (int i = 0; i < N; i++) {                                                                                   // for n = 1:N
                if (alpha.getEntry(0,i) > 0) {                                                                                     //if alpha(n) > 0
                    DenseMatrix MnUnfolding = matlab.Unfold(M, NwayDV, i);                                                       // Mn = Unfold(M,Nway,n);
                    if (solX.getEntry(0, i) > 0) {                                                                          //if solX(n)

                        X.set(i, (DenseMatrix) ml.utils.Matlab.mtimes(MnUnfolding, Yt.get(i)));                             //   X{n} = Mn*Yt{n};
                    }
                    solX.setEntry(0, i, 1);                                                                                 //SolX(n) = 1;

                    DenseMatrix Xt = (DenseMatrix) X.get(i).transpose();
                    Xsq.set(i, (DenseMatrix) ml.utils.Matlab.mtimes(Xt, X.get(i)));                                         // Xsq{n} = X{n}'*X{n};

                    Jama.Matrix JamaXsq = matlab.LAML2JAMAMatrix(Xsq.get(i));
                    Jama.Matrix Xsqpinv = matlab.Jamapinv(JamaXsq);

                    DenseMatrix pinvXsq = matlab.JAMA2LAMLMatrix(Xsqpinv);
                    DenseMatrix matrixMul = (DenseMatrix) ml.utils.Matlab.mtimes(Xt, MnUnfolding);
                    Y.set(i, (DenseMatrix) ml.utils.Matlab.mtimes(pinvXsq, matrixMul));


                 //
                     
                    Yt.set(i, (DenseMatrix) Y.get(i).transpose());
 System.out.println();
                }
            }

            //% update M++++
            DenseMatrix XtimesY = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(0), Y.get(0));
            Mn.set(0, matlab.Fold(XtimesY, NwayDV, 0));                                                 // Mn = Fold(X{1}*Y{1},Nway,1);
            ArrayList<Double> estimationKnownDiff = matlab.EstimationKnownDiff(Mn.get(0), data, known);
            DenseVector estimationKnownDense = matlab.ArrayListtoDenseVector(estimationKnownDiff);
            res.setEntry(0, 0, ml.utils.Matlab.norm(estimationKnownDense));

            Tensor tensorMul = Mn.get(0).TensorMul(Mn.get(0), alpha.getEntry(0,0));
            M = tensorMul;
System.out.println();
            for (int i = 1; i < N; i++) {                                                                          //    for n = 2:N
                if (alpha.getEntry(0,i) > 0) {                                                                     //    if alpha(n) > 0
                    XtimesY = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(i), Y.get(i));
                    Mn.set(i, matlab.Fold(XtimesY, NwayDV, i));                                                 // Mn = Fold(X{1}*Y{1},Nway,1);
                    estimationKnownDiff = matlab.EstimationKnownDiff(Mn.get(i), data, known);
                    estimationKnownDense = matlab.ArrayListtoDenseVector(estimationKnownDiff);
                    res.setEntry(0, i, ml.utils.Matlab.norm(estimationKnownDense));

                    tensorMul = Mn.get(i).TensorMul(Mn.get(i), alpha.getEntry(0,i));
                    M = M.TensorAdd(M, tensorMul);                                                              //    M = M+alpha(n)*Mn;


                }
            }
//% pass the true tensor M for evaluation
            M.copyTensorObserved(data, known);

            double TotalRes0 = TotalRes;
            TotalRes = 0;                                                                                        // TotalRes = 0;
            for (int i = 0; i < N; i++) {                                                                           //for n = 1:N
                if (alpha.getEntry(0, i) > 0) {                                                                  //if alpha(n) > 0
                    TotalRes = TotalRes + (res.getEntry(0, i) * res.getEntry(0, i));                             //TotalRes = TotalRes+res(n)^2;
                }                                                                                                //end
            }                                                                                                    //end

            DenseMatrix ratio = new DenseMatrix(1, 3);

            for (int i = 0; i < N; i++)
                ratio.setEntry(0, i, res.getEntry(0, i) / res0.getEntry(0, i));                  //ratio = res./res0;

            for (int i = 0; i < N; i++)
                reschg.setEntry(0, i, Math.abs(1 - ratio.getEntry(0, i)));                      //reschg = abs(1-ratio);
IO io = new IO();


//if (rank_inc_num > 0) {   

// if rank_inc_num > 0
                for (int i = 0; i < N; i++) {                                                                                // for n = 1:N
                    //if (alpha.getEntry(0, i) > 0) {                                                                          // if alpha(n) > 0
                      // if (coreNway.getEntry(0, i) < rank_max.getEntry(0, i) && reschg.getEntry(0, i) < reschg_tol) {
               //if (coreNway.getEntry(0, i) < rank_max.getEntry(0, i)){            
System.out.println("RANK INC CALL");
// if coreNway(n) < rank_max(n) && reschg(n) < reschg_tol
                          
                        if(isTest) rank_inc_adaptive(M,i,NwayDV,X,Y); 
                        else rank_inc_adaptiveExample(M,i,NwayDV,X,Y);                              //rank_inc_adaptive();
                       //rank_2(M,i,NwayDV,X,Y); 
                      // }
                  }
              //  }
//           / }



System.out.println();
 //rank_inc_adaptive(M,0,NwayDV,X,Y);
//% adaptively update weight
            if (false) {
                for (int i = 0; i < N; i++)
                    alpha.setEntry(0,i, 1 / (res.getEntry(0, i) * res.getEntry(0, i)));                   //alpha = 1./(res.^2);
                for (int i = 0; i < N; i++)
                    alpha.setEntry(0,i, alpha.getEntry(0,i) / matlab.sum(alpha));                     //alpha = alpha/sum(alpha);

            }

            //% --- diagnostics, reporting, stopping checks ---

            relerr1 = Math.abs(TotalRes - TotalRes0) / (TotalRes0 + 1);                                     //relerr1 = abs(TotalRes-TotalRes0)/(TotalRes0+1);


            DenseMatrix alphaScale = new DenseMatrix(1, 3);
            for (int i = 0; i < N; i++)
                alphaScale.setEntry(0, i, alpha.getEntry(0,i) * res.getEntry(0, i));

            relerr2 = matlab.sum(alphaScale) / nrmb;                                                             //  relerr2 = sum(alpha.*res)/nrmb;


            // % check stopping criterion
            boolean crit = relerr1 < tol;                                                                              //crit = relerr1<tol;

            if (crit) nstall = nstall + 1;
            else nstall = 0;
            if (nstall >= 3 || relerr2 < tol) break;

            //  TODO   if toc(start_time)>maxT; break; end;


            for (int i = 0; i < N; i++) {
                X0.set(i, matlab.DenseMatrixCopy(X.get(i)));
                Y0.set(i, matlab.DenseMatrixCopy(Y.get(i)));

            }
            M0 = M.Copy();
            res0 = res;
;

        }
        
        // ΤΕΡΟΟΟΣ
       IO io = new IO();

        
        DenseMatrix XtimesY1 = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(0), Y.get(0));
        DenseMatrix XtimesY2 = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(1), Y.get(1));
        DenseMatrix XtimesY3 = (DenseMatrix) ml.utils.Matlab.mtimes(X.get(2), Y.get(2));

        
        Tensor folding1 = matlab.Fold(XtimesY1, NwayDV, 0);
        Tensor folding2 = matlab.Fold(XtimesY2, NwayDV, 1);
        Tensor folding3 = matlab.Fold(XtimesY3, NwayDV, 2);

            folding1.copyTensorObserved(data, known);
            folding2.copyTensorObserved(data, known);
            folding3.copyTensorObserved(data, known);
        for(int i = 0; i<folding1.getTensor()[0][0].length; i++){
           
            DenseMatrix kMatrix1 = folding1.getKMatrix(i);
             
             int matq = i+1;
             
        ArrayList<Double> DH = matlab.DehankelizationR(kMatrix1, W, overlap);
    io.writeArrayList2File(DH,pathRes+ "_"+matq+"_rec1.csv");
    
          DenseMatrix kMatrix2 = folding2.getKMatrix(i);
        ArrayList<Double> DH2 = matlab.DehankelizationR(kMatrix2, W, overlap);
    io.writeArrayList2File(DH2, pathRes+"_"+matq+"_rec2.csv");
        
    
          DenseMatrix kMatrix3 = folding3.getKMatrix(i);
        ArrayList<Double> DH3 = matlab.DehankelizationR(kMatrix3, W, overlap);
    io.writeArrayList2File(DH3,pathRes+ "_"+matq+"_rec3.csv");
        }


ArrayList<Tensor> result  = new ArrayList<Tensor>();
result.add(folding1); result.add(folding2) ;result.add(folding3);
    return result;
    }






    public void rank_inc_adaptive(Tensor t,int i, DenseVector NwayDV,ArrayList<DenseMatrix> X,ArrayList<DenseMatrix> Y) {
       // System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< rank_inc_ad "+i);

double max = t.maxEntry();
 double min =  t.minEntry();
  Random ran = new Random();
       
       System.out.println();
        Matlab matlab = new Matlab();
        IO io = new IO();
        Yt.set(i, (DenseMatrix) Y.get(i).transpose());


       QRDecomposition QR = new QRDecomposition(matlab.LAML2JAMAMatrix(Yt.get(i)));
        Jama.Matrix H = QR.getH();
        Jama.Matrix q = QR.getQ();
        Jama.Matrix r = QR.getR();
        
     
        
        Jama.Matrix qt = q.transpose();
        DenseMatrix Q = matlab.JAMA2LAMLMatrix(q);
        Jama.Matrix QRm = q.times(r);
        DenseMatrix Qt = (DenseMatrix) Q.transpose();
        
;
        
        System.out.println();

        DenseMatrix merge;

//        for (int ii = 0; ii < rank_inc.getEntry(0, i); ii++) { // for ii = 1:rank_inc(n)
//            ArrayList<Double> rdnxList;
//            DenseMatrix rdnx = new DenseMatrix((int) coNway.getEntry(0,i), 1);
//
//            for (int iii=0; iii<rdnx.getRowDimension(); iii++){
//                rdnx.setEntry(iii, 0, min + (max - min) * ran.nextDouble());
//				
//				// rdnx.setEntry(iii, 0, min + (max - min) * ran.nextDouble());
//            }


         DenseMatrix rdnx;
         //
//       double max = M.maxEntry(); 
//      double  min = M.minEntry();
//      double range = max - min + 1; 
//      for (int zz = 0; zz < coNway.getEntry(0,i); zz++) {
//                rdnx.setEntry(zz, 0, (Math.random() * range) + min);
//               
//    }
         
         
         //= new DenseMatrix((int) coNway.getEntry(0,i), 1);
        if(i==0) rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path, "rdnx1.csv");
        else if(i==1)  rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path,"rdnx2.csv");
        else rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path, "rdnx3.csv");
        
            DenseMatrix mul1 = (DenseMatrix) ml.utils.Matlab.mtimes(Q.transpose(), rdnx);
            DenseMatrix mul2 = (DenseMatrix) ml.utils.Matlab.mtimes(Q, mul1);

            rdnx = (DenseMatrix) rdnx.minus(mul2);
            rdnx = (DenseMatrix) rdnx.times(1 / ml.utils.Matlab.norm(rdnx));

            merge = matlab.Merge(Q, rdnx);
            Y.set(i, (DenseMatrix) merge.transpose());
            Yt.set(i, merge);
            Y0.set(i, (DenseMatrix) merge.transpose());


       

        coreNway.setEntry(0,i, (int) (coreNway.getEntry(0,i) + rank_inc.getEntry(0, i)));
        if (coreNway.getEntry(0,i) >= rank_max.getEntry(0, i)) {
            rank_inc_num = rank_inc_num - 1;
        }
        if (rank_inc_num == 0) {
            nstall = 0;
        }
        DenseMatrix MnUnfolding = matlab.Unfold(M, NwayDV, i);
        X.set(i, (DenseMatrix) ml.utils.Matlab.mtimes(MnUnfolding, Y.get(i).transpose()));
        X0.set(i, X.get(i));
        solX.setEntry(0, i, 0);
  
    }



    public void rank_inc_adaptiveExample(Tensor t,int i, DenseVector NwayDV,ArrayList<DenseMatrix> X,ArrayList<DenseMatrix> Y) {
       // System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< rank_inc_ad "+i);

double max = t.maxEntry();
 double min =  t.minEntry();
  Random ran = new Random();
       
       System.out.println();
        Matlab matlab = new Matlab();
        IO io = new IO();
        Yt.set(i, (DenseMatrix) Y.get(i).transpose());


       QRDecomposition QR = new QRDecomposition(matlab.LAML2JAMAMatrix(Yt.get(i)));
        Jama.Matrix H = QR.getH();
        Jama.Matrix q = QR.getQ();
        Jama.Matrix r = QR.getR();
        
     
        
        Jama.Matrix qt = q.transpose();
        DenseMatrix Q = matlab.JAMA2LAMLMatrix(q);
        Jama.Matrix QRm = q.times(r);
        DenseMatrix Qt = (DenseMatrix) Q.transpose();
        
;
        
        System.out.println();

        DenseMatrix merge;

//        for (int ii = 0; ii < rank_inc.getEntry(0, i); ii++) { // for ii = 1:rank_inc(n)
//            ArrayList<Double> rdnxList;
//            DenseMatrix rdnx = new DenseMatrix((int) coNway.getEntry(0,i), 1);
//
//            for (int iii=0; iii<rdnx.getRowDimension(); iii++){
//                rdnx.setEntry(iii, 0, min + (max - min) * ran.nextDouble());
//				
//				// rdnx.setEntry(iii, 0, min + (max - min) * ran.nextDouble());
//            }


         DenseMatrix rdnx;
         //
//       double max = M.maxEntry(); 
//      double  min = M.minEntry();
//      double range = max - min + 1; 
//      for (int zz = 0; zz < coNway.getEntry(0,i); zz++) {
//                rdnx.setEntry(zz, 0, (Math.random() * range) + min);
//               
//    }
//         Path path = Paths.get(TCJGit.class.getResource(".").toURI());
//System.out.println(path.getParent()); 
         
         //= new DenseMatrix((int) coNway.getEntry(0,i), 1);
        if(i==0) rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path, "rdnx1.csv");
        else if(i==1)  rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path, "rdnx2.csv");
        else rdnx = io.readDenseMatrix((int)coNway.getEntry(0,i), 1, path, "rdnx3.csv");
        
            DenseMatrix mul1 = (DenseMatrix) ml.utils.Matlab.mtimes(Q.transpose(), rdnx);
            DenseMatrix mul2 = (DenseMatrix) ml.utils.Matlab.mtimes(Q, mul1);

            rdnx = (DenseMatrix) rdnx.minus(mul2);
            rdnx = (DenseMatrix) rdnx.times(1 / ml.utils.Matlab.norm(rdnx));

            merge = matlab.Merge(Q, rdnx);
            Y.set(i, (DenseMatrix) merge.transpose());
            Yt.set(i, merge);
            Y0.set(i, (DenseMatrix) merge.transpose());


       

        coreNway.setEntry(0,i, (int) (coreNway.getEntry(0,i) + rank_inc.getEntry(0, i)));
        if (coreNway.getEntry(0,i) >= rank_max.getEntry(0, i)) {
            rank_inc_num = rank_inc_num - 1;
        }
        if (rank_inc_num == 0) {
            nstall = 0;
        }
        DenseMatrix MnUnfolding = matlab.Unfold(M, NwayDV, i);
        X.set(i, (DenseMatrix) ml.utils.Matlab.mtimes(MnUnfolding, Y.get(i).transpose()));
        X0.set(i, X.get(i));
        solX.setEntry(0, i, 0);
  
    }


}
    
