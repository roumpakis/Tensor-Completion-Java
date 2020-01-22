/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

import java.util.ArrayList;
import la.matrix.DenseMatrix;
import tensorcompletion.Tensor;

/**
 *
 * @author Roubakas
 */
public class TCJTest {
    
    
    public static void Test( int W, double overlap ){
        final String dir = System.getProperty("user.dir");
        System.out.println("current dir = " + dir);
        String path = dir+"\\data\\";
        Matlab matlab = new Matlab();
        IO io = new IO();
        ArrayList<Double> D7InMiss = io.ReadVars(path, "D7InMiss.csv");
        ArrayList<Double> D8InMiss = io.ReadVars(path, "D8InMiss.csv");
        ArrayList<Double> D9InMiss = io.ReadVars(path, "D9InMiss.csv");
       // ArrayList<Double> D10InMiss = io.ReadVars(path, "D10InMiss.csv");

        DenseMatrix D7InMissM = matlab.HankelizationR(D7InMiss, W, overlap) ;
        DenseMatrix D8InMissM = matlab.HankelizationR(D8InMiss, W, overlap) ;
        DenseMatrix D9InMissM = matlab.HankelizationR(D9InMiss, W, overlap) ;
       // DenseMatrix D10InMissM = matlab.HankelizationR(D10InMiss, W, overlap) ;
        
        
    ArrayList<DenseMatrix> DList = new ArrayList<DenseMatrix>();
    DList.add(D7InMissM); DList.add(D8InMissM);
    DList.add(D9InMissM); 
    //DList.add(D10InMissM);
    
     Tensor T = matlab.Tensorization(DList);
        ArrayList<Double> data = T.TensorObservedElements();
        ArrayList<TensorIndex> known = T.TensorObservedIndecies();
     TMacPar TC = new TMacPar();
     ArrayList<DenseMatrix> X = new ArrayList<DenseMatrix>();
     ArrayList<DenseMatrix> Y = new ArrayList<DenseMatrix>();
     
        DenseMatrix X1 = io.readDenseMatrix(64, 8, path, "X_1.csv");
        DenseMatrix X2 = io.readDenseMatrix(155, 17, path, "X_2.csv");
        DenseMatrix X3 = io.readDenseMatrix(3, 1, path, "X_3.csv");
        X.add(X1); X.add(X2); X.add(X3);
        
        DenseMatrix Y1 = io.readDenseMatrix(8, 465, path, "Y_1.csv");
        DenseMatrix Y2 = io.readDenseMatrix(17, 192, path, "Y_2.csv");
        DenseMatrix Y3 = io.readDenseMatrix(1, 9920, path, "Y_3.csv");
        Y.add(Y1); Y.add(Y2); Y.add(Y3);
        
        ArrayList<Tensor> Trec = TC.TMAC(T,X,Y, known, data , true,W,overlap);
     System.out.println();
     
        
        
    }
    
}
