/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package TC;


import Utilities.IO;
import tensorcompletion.Tensor;
import Utilities.Matlab;

import Utilities.TCJTest;
import Utilities.TMacPar;
import Utilities.TensorIndex;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import la.matrix.DenseMatrix;

/**
 *
 * @author Roubakas
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static Matlab matlab = new Matlab();

    public static void main(String[] args) throws IOException {
        
        int W = 64;
        double overlap = 0.5;
        TCJTest.Test(W, overlap);


    }   
    



public static void TMaacParTest(HashMap<Integer, ArrayList<Double>> waterData, int W, double overlap){

                Matlab Matlab = new Matlab();
                IO io = new IO();
   


                int max = 10; 
                int min = 0; 
                int range = max - min + 1;
                

                 
                 
                 ArrayList<DenseMatrix> allDMs = new ArrayList<DenseMatrix>();
                 
        Set<Integer> keySet = waterData.keySet();
        for(int i=0; i<keySet.size(); i++){
            ArrayList<Double> stream = waterData.get(i);
            DenseMatrix HankeledStream =  Matlab.HankelizationR(stream,W,overlap);
            allDMs.add(HankeledStream);
        }

           
                    // Tensorization -  Create a tensor using HankeledStreams as slices
                  Tensor T = matlab.Tensorization(allDMs);

 Tensor TInit = T.Copy();
     Tensor TMiss= matlab.MeasurementUndersaple(T, 0.5);
     ArrayList<Double> data = TMiss.TensorObservedElements();
     ArrayList<TensorIndex> known = TMiss.TensorObservedIndecies();
     double maxEntry = TInit.maxEntry();
    double minEntry = TInit.minEntry();
    ArrayList<DenseMatrix> X  = new ArrayList<DenseMatrix>();
    ArrayList<DenseMatrix> Y = new ArrayList<DenseMatrix>();
    
    
    
    
    
        int I = TInit.getTensor().length;
        int J = TInit.getTensor()[0].length;
        int K = TInit.getTensor()[0][0].length;


        // oi Pragmatikes diastaseis tou Tensor
        DenseMatrix Nway = new DenseMatrix(1, 3);
        Nway.setEntry(0, 0, I);
        Nway.setEntry(0, 1, J);
        Nway.setEntry(0, 2, K);

// estimated ranks of all mode matricizations defaults ranks = 10
        DenseMatrix coreNway = new DenseMatrix(1, 3);
        coreNway.setEntry(0, 0, 10);
        coreNway.setEntry(0, 1, 10);
        coreNway.setEntry(0, 2,10);
/* coreNway =  estimatedRank1.....*/
        //  the co Nway
        DenseMatrix coNway = new DenseMatrix(1, 3);
        for (int i = 0; i < 3; i++) {
            double integer = matlab.prod(Nway) / Nway.getEntry(0, i);
            coNway.setEntry(0, i, integer);
        }
        

int N = 3;
   
            for (int i = 0; i < N; i++) {
            int size1 = (int) Nway.getEntry(0, i);
            int size2 = (int) coreNway.getEntry(0, i);
          X.add(i, (DenseMatrix) ml.utils.Matlab.randn(size1, size2));
        
            
            //X.add(i,createRandomDenseMatrixRanged(size1,size2,  TInit.meanEntry(),  TInit.meanEntry()));
           //X.add(i,createRandomDenseMatrixRanged(size1,size2,  minEntry,  maxEntry));
            
            }

        for (int i = 0; i < N; i++) {
            int size3 = (int) coreNway.getEntry(0, i);
            int size4 = (int) coNway.getEntry(0, i);

            Y.add(i, (DenseMatrix) ml.utils.Matlab.randn(size3, size4));   
            //Y.add(i,createRandomDenseMatrixRanged(size3,size4,  TInit.meanEntry(),  TInit.meanEntry()));
                //   Y.add(i,createRandomDenseMatrixRanged(size3,size4,  minEntry,  maxEntry));//  Y{n} = randn(coreNway(n),coNway(n));
        }
    
    
   
    TMacPar TC = new TMacPar();
    ArrayList<Tensor> recL = TC.TMAC(TMiss, X, Y, known, data, false,W,overlap);
    ArrayList<DenseMatrix> reconstructedHankeledStreams = new ArrayList<DenseMatrix>();
    
  for(int j=0; j<recL.size(); j++) {
      Tensor rec = recL.get(j);
    for(int i=0; i<K; i++){
        reconstructedHankeledStreams.add(rec.getKMatrix(i));
    }
    
    ArrayList<ArrayList<Double>> reconstrucedStreams = new ArrayList<ArrayList<Double>>();
    for(int i=0; i<reconstructedHankeledStreams.size(); i++){
        reconstrucedStreams.add( matlab.DehankelizationR(reconstructedHankeledStreams.get(i),W,overlap));
    }
 }

              System.out.println();
}

public static DenseMatrix createRandomDenseMatrixRanged(int rows, int cols, double min, double max){
    DenseMatrix res = new DenseMatrix(rows,cols);
     double range = max - min + 1; 
     
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            res.setEntry(i, j, (Math.random() * range) + min);
        }
    }
    return res;
}




}
    
//}