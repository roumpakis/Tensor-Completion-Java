/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

import java.util.ArrayList;
import java.util.List;
import la.matrix.DenseMatrix;
import la.vector.Vector;

/**
 *
 * @author Roubakas
 */
public class Hankel {
    
    ArrayList<Double> data;
    IO io;
    Matlab matlab;
    DenseMatrix HankeledData;
    
   public Hankel(ArrayList<Double> data){
        this.data = data;
        this.io = new IO();
        this.matlab = new Matlab();
        
    }
    
    public DenseMatrix Hankelization(int windowSize,int step){
          int numSamples = data.size();
    int start_point = 0;
    int end_point = start_point+windowSize-1;
   int numWindows = Math.floorDiv(numSamples-start_point-windowSize+1 , step+1);
  //  int numWindows =(numSamples-start_point-windowSize+1)/ step+1;
   // numWindows = numWindows-1;
    DenseMatrix data_mtx = new DenseMatrix(numWindows,windowSize);
    for(int t =0; t<numWindows; t++){
        
        List<Double> subList = data.subList(start_point, end_point);
        for(int j = 0; j<subList.size(); j++){
            
            data_mtx.setEntry(t, j,subList.get(j));
            
        
        }
        start_point = start_point+step;
        end_point = start_point+windowSize;
    }
    
    return data_mtx;
    }
    
     public DenseMatrix HankelizationIDXMatrix(int windowSize,int step){
          int numSamples = data.size();
    int start_point = 0;
    int end_point = start_point+windowSize-1;
    int numWindows = (numSamples-start_point-windowSize+1)/step+1;
    
    DenseMatrix idx = new DenseMatrix(numWindows,windowSize);
    for(int t =0; t<numWindows; t++){
        List<Double> subList = data.subList(start_point, end_point);
        for(int j = 0; j<subList.size(); j++){
            
            idx.setEntry(t, j,start_point+j);
            
        
        }
        start_point = start_point+step;
        end_point = start_point+windowSize;
    }
    
    return idx;
    }
    public ArrayList <Double> Dehankelization (DenseMatrix H, int windowSize,int step){
        int rows = H.getRowDimension();
        int cols = H.getColumnDimension();
        ArrayList<Double> res = new ArrayList<>();
        
        for(int j = 0; j<cols; j++) res.add(H.getEntry(0,j));
        System.out.println(" size "+res.size());
        for(int i = 1; i<rows; i++){
          
            for(int j = cols - step; j<cols; j++){
                System.out.println("( "+i+","+j+" )");
                res.add(H.getEntry(i,j));
            }
        }
        
        
        return res;
    }
    
    
}
