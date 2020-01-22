package tensorcompletion;


import Utilities.Matlab;
import Utilities.TensorIndex;

import java.awt.font.TextAttribute;
import java.lang.reflect.Array;
import java.security.Timestamp;
import java.util.ArrayList;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Random;

import la.decomposition.SingularValueDecomposition;
import la.matrix.DenseMatrix;
import la.vector.DenseVector;

/**
 * Created by roubakas on 4/12/2017.
 */

public class Tensor {
    private double [][][] tensor;
    ArrayList<TensorIndex> Unknown  = new ArrayList<>();
    ArrayList<Double> data = new ArrayList<>();
HashMap<Integer, HashMap< Integer, Integer> >UknownMap = new HashMap<Integer, HashMap<Integer,Integer> >();;



    public Tensor(int timestamps, int devices, int modalities){
for(int i = 0; i<timestamps; i++)

        tensor  = new double [timestamps][devices][modalities];
//        System.out.println("<<<<<<<<<< new Tensor Created! >>>>>>>>>>");
    }







    public void addElement(int indexX,int indexY, int indexZ,double element){
        
//System.out.println(indexX+","+indexY+","+indexZ);
    }


    @Override
    public String toString() {
        String res="";
        for(int i = 0;i<tensor.length; i++) {
            for (int j = 0; j<tensor[i].length; j++) {
                for (int k =0; k<tensor[i][j].length; k++) {
                    res = res + " ("+i+","+j+","+k+")"+tensor[i][j][k];
                }
                res = res+"\n";
            }
        }
        return res;
    }

    public double [][] TensorUnfolding(int dimension){

        int X = tensor.length;
        int Y = tensor[0].length;
        int Z = tensor[0][0].length;
        // System.out.println("<<<<<< X = "+X+" , Y = "+Y+" , Z = " +Z +" >>>>>>>>>");
        int resX = 0;
        int resY = 0;
        double [][] res;

        if(dimension == 1){
            resX = 0;
            resY = 0;
            res = new double [X][Y*Z];
            for(int i =0; i<X; i++){
                for(int j =0; j<Y; j++){
                    for(int k =0; k<Z;k++){
//                    System.out.print("("+i+","+j+","+k+")");
//                    System.out.print(resX+","+resY+"   ");
                        res[resX][resY] = tensor[i][j][k];
                        resY++;
                    }
                }
                resY =0;
                resX++;
//            System.out.println();
            }
        }
        else if(dimension == 2){
            resX = 0;
            resY = 0;
            res = new double [Y][X*Z];
            for(int j =0; j<Y; j++){

                for(int k =0; k<Z;k++){
                    for(int i =0; i<X; i++){
//                    System.out.print("("+i+","+j+","+k+")");
//                    System.out.print(resX+","+resY+"   ");

                        res[resX][resY] = tensor[i][j][k];
                        resY++;
                    }



                }
                resX++;
                resY++;
//            resX  = 0;
                resY = 0;
                System.out.println();
            }
        }

        else if(dimension == 3){
            res = new double [Z][X*Y];
            for(int k =0; k<Z;k++){
                for(int i =0; i<X; i++){
                    for(int j =0; j<Y; j++){

//                    System.out.print("("+i+","+j+","+k+")");
//                System.out.print(resX+","+resY+"   ");
                        res[resX][resY] = tensor[i][j][k];
                        resY++;

                    }

                }
                resY =0;
                resX++;
                System.out.println();
            }

        }
        else{
            res=null;
            System.out.println("!!!!!!!!!!!! ERROR UNFOLDING MODE !!!!!!!!!!");
        }



        return res;
    }

    public Tensor Folding(DenseMatrix Matrix, DenseMatrix Nway, int dimension){
        int timestamps = (int)Nway.getEntry(0,0);
        int devices = (int)Nway.getEntry(0,1);
        int modality = (int)Nway.getEntry(0,2);

        //System.out.println("<<<<<< X = "+timestamps+" , Y = "+devices+" , Z = " +modality +" >>>>>>>>>");
        Tensor tensor = new Tensor(timestamps,devices,modality);

        if(dimension == 1){
//System.out.println("Folding: mode1");
            for(int i = 0; i<Matrix.getRowDimension(); i++){
                int tensorDevice = 0;
                int tensorModality = 0;
                for(int j = 0; j<Matrix.getColumnDimension(); j++){
                    // otan eisagoume ola ta modalities autou tou device, pame sto epomeno ksekinontas apo to prwto modality
//
//System.out.println("["+i+","+tensorDevice+","+tensorModality+"] = "+Matrix.getEntry(i,j));
                    tensor.setEntry(i,tensorDevice,tensorModality,Matrix.getEntry(i,j));
                    tensorModality++;

                    if(tensorModality%modality==0){
                        tensorModality=0;
                        tensorDevice++;
                    }
                }
                tensorDevice = 0;
                tensorModality = 0;
            }
        }




        if(dimension == 2){
            //   System.out.println("Folding: mode2");
            for(int i = 0; i<Matrix.getRowDimension(); i++){
                int tensorTimestamp = 0;
                int tensorModality = 0;
                for(int j = 0; j<Matrix.getColumnDimension(); j++){
                    // otan eisagoume ola ta modalities autou tou timestamp, pame sto epomeno ksekinontas apo to prwto modality

                  //  System.out.println("["+ tensorTimestamp+","+i+","+tensorModality+"] = "+Matrix.getEntry(i,j));
                    tensor.setEntry(tensorTimestamp,i,tensorModality,Matrix.getEntry(i,j));
                    tensorTimestamp++;

                    if(tensorTimestamp%timestamps==0){
                        tensorTimestamp=0;
                        tensorModality++;
                    }
                }
            }
        }


        if(dimension == 3){
            // System.out.println("Folding: mode3");

            for(int i = 0; i<Matrix.getRowDimension(); i++){
                int tensorTimestamp = 0;
                int tensorDevice = 0;
                for(int j = 0; j<Matrix.getColumnDimension(); j++){
                    // otan eisagoume ola ta modalities autou tou timestamp, pame sto epomeno ksekinontas apo to prwto modality

//                System.out.println("["+ tensorTimestamp+","+tensorDevice+","+i+"] = "+Matrix.getEntry(i,j));
                    tensor.setEntry(tensorTimestamp,tensorDevice,i,Matrix.getEntry(i,j));
                    tensorDevice++;

                    if(tensorDevice%devices==0){
                        tensorDevice=0;
                        tensorTimestamp++;
                    }
                }
            }
        }
System.out.println();
        return tensor;
    }



    // return the fil ratio based on the non zero elements
    public float tensorFilRatio(){
        int sum = 0;
        int N = 0;
        for(int i = 0; i<tensor.length; i++){
            for(int j =0; j<tensor[i].length; j++){
                for(int k  = 0; k<tensor[i][j].length; k++){
                    if(tensor[i][j][k]!=0) sum++;
                    N++;
                }
            }
        }
        float res = (sum * 100.0f)/N;

        System.out.println("N =   "+N+"   sum = "+sum+" Fill Ration = "+res+"%");
        return res;
    }

    //printarismata
    public void showDevicMeasurements(int device){
        if(device < 0 || device > 4){
            System.out.println("ERROR! Choose a device between [0,4]");
        }
        else{
            int sum = 0;
            for(int i = 0; i<tensor.length; i++){
                for(int j =0; j<tensor[i][device].length; j++){
                    System.out.print(" "+tensor[i][device][j]);
                }
                System.out.println();
                sum++;
            }
            System.out.println("<><><>><<><><>><><>< "+sum);
        }

    }

    public void showModalityMeasurements (int modality) {

        if (modality < 0 || modality > 8) {
            System.out.println(" Choose a modality between [0,8]");
        } else {
            int sum = 0;
            for (int i = 0; i < tensor.length; i++) {
                for (int j = 0; j < tensor[i].length; j++) {
                    System.out.print(" " + tensor[i][j][modality]);

                }
                System.out.println();
                sum++;
            }
            System.out.println("<><><>><<><><>><><>< " + sum);

        }
    }


    public void showTimestampMeasurements (int timestamp){

        if(timestamp < 0 || timestamp >= tensor.length){
            System.out.println(" Choose a timestamp between [0,"+ tensor.length+"]");
        }else{
            int sum = 0;
            for(int i =0; i<tensor.length; i++){
                for(int j =0; j<tensor[i].length; j++){
                    if(tensor[i][j][0]!= 0){
                        //    for(int k =0; k<tensor[i][j].length; k++){
                        System.out.print(" "+tensor[i][j][0]);
                        // }
                    }
                }
                System.out.println();
                sum++;
            }
            System.out.println("<><><>><<><><>><><>< "+sum);

        }

    }
    public ArrayList<TensorIndex> TensorObservedIndecies() {
        System.out.println("Tensor Observed Colled!!!");
        ArrayList<TensorIndex> res = new ArrayList<TensorIndex>();
        for (int i = 0; i < tensor.length; i++) {
            for (int j = 0; j < tensor[i].length; j++) {
                for (int z = 0; z < tensor[i][j].length; z++) {
                    if (tensor[i][j][z]!=0.0) {
                        TensorIndex tensorIndex = new TensorIndex(i, j, z);
                        res.add(tensorIndex);
                    }
                }
            }
        }
        System.out.println("Tensor Observed Returned!!!");
        return res;
    }


    public ArrayList<Double> TensorObservedElements(){
        System.out.println("Tensor Observed Elements Called!!!");
        ArrayList<Double> res = new ArrayList<Double>();
        for (int i = 0; i < tensor.length; i++) {
            for (int j = 0; j < tensor[i].length; j++) {
                for (int z = 0; z < tensor[i][j].length; z++) {
                    if (tensor[i][j][z]!=0.0) {
                        //System.out.println("Observed Element !2!@#!@#@@@SSKSKKS "+tensor[i][j][z]+" total = "+res.size());
                        res.add(tensor[i][j][z]);
                    }
                }
            }
        }
        System.out.println("Tensor Observed Elements Returned!!!");
        return res;
    }

public void setEntry(int x,int y,int z, double value){
  //  System.out.println("("+x+","+y+","+z+")"+" = "+value);
    this.tensor[x][y][z] = value;
    
}

    public Tensor copyTensorObserved( ArrayList<Double> observedElements, ArrayList<TensorIndex> tensorIndices){

        for(int i = 0; i<tensorIndices.size(); i++){
           // System.out.println(i+" size: "+tensorIndices.size());
           TensorIndex tensorIdx = tensorIndices.get(i);
           int x = tensorIdx.getX();
           int y = tensorIdx.getY();
           int z =  tensorIdx.getZ();
           if(x==-1 && y==-1 && z==-1) System.out.println("EEEEEEEEEEEEEEEi "+i);
           double value = observedElements.get(i);
           
            this.setEntry(x,y,z,value);
        }
        return this;
    }


    public Tensor Copy(){
        int X = tensor.length;
        int Y = tensor[0].length;
        int Z = tensor[0][0].length;

        Tensor newTensor = new Tensor(X,Y,Z);
        for(int a = 0; a<X; a++){
            for(int b = 0; b<Y; b++){
                for(int c = 0; c<Z; c++){
                    newTensor.setEntry(a,b,c,getTensor()[a][b][c]);
                }
            }
        }
        return newTensor;
    }

    public Tensor TensorMul(Tensor t, double times){
        for(int i = 0; i<tensor.length; i++){
            for(int j  = 0; j<tensor[0].length; j++){
                for(int k = 0; k<tensor[0][0].length; k++){
                    t.getTensor()[i][j][k]  = t.getTensor()[i][j][k] * times;
                }
            }
        }
        return t;
    }
    public Tensor TensorAdd(Tensor t1,Tensor t2){
        for(int i = 0; i<tensor.length; i++){
            for(int j  = 0; j<tensor[0].length; j++){
                for(int k = 0; k<tensor[0][0].length; k++){
                    t1.getTensor()[i][j][k]  = t1.getTensor()[i][j][k] + t2.getTensor()[i][j][k];
                }
            }
        }
        return t1;
    }


    // Setters & Geetters
    public void setTensor(double[][][] tensor) {
        this.tensor = tensor;
    }

    public double[][][] getTensor() {
        return tensor;
    }



    public TensorIndex mapLinearTI(int linear){

        int X = tensor.length;
        int Y = tensor[0].length;
        int Z = tensor[0][0].length;

        int counter = 1;
       


        for(int k = 0; k<Z; k++){
            for(int j = 0; j<Y; j++){
                for(int i = 0 ; i<X; i++){

                    if(counter==linear) {
                     TensorIndex res = new TensorIndex(i,j,k);
                     return res;
                       
                    }
                    counter++;
                }
            }
        }

        return null;
    }

    public Tensor TensorDiff(Tensor arg){
        double[][][] doubles = this.getTensor();
        int X = doubles.length;
        int Y = doubles[0].length;
        int Z = doubles[0][0].length;
        Tensor res = new Tensor(X,Y,Z);

    for(int  i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                res.addElement(i,j,k, doubles[i][j][k]
                - arg.getTensor()[i][j][k]);
            }
        }
    }

    return res;
    }

    public DenseVector Tensor2DenseVector(){


        double[][][] doubles = this.getTensor();
        int X = doubles.length;
        int Y = doubles[0].length;
        int Z = doubles[0][0].length;
        DenseVector res = new DenseVector(X*Y*Z);
int counter = 0;

        for(int  i = 0; i<X; i++){
            for(int j = 0; j<Y; j++){
                for(int k = 0; k<Z; k++){
                  res.set(counter,doubles[i][j][k]);
                    counter++;
                }
            }
        }


        return res;
    }

public Tensor CopyRealData(){
    double[][][] doubles = this.getTensor();
    int X = doubles.length;
    int Y = doubles[0].length;
    int Z = doubles[0][0].length;
    int realRows = 0;

    for(int i = 0; i<X; i++){
    if(
             doubles[i][0][0]!=0 || doubles[i][0][1]!=0  || doubles[i][0][2]!=0
            || doubles[i][0][3]!=0 || doubles[i][0][4]!=0 || doubles[i][0][5]!=0
            || doubles[i][0][6]!=0 || doubles[i][0][7]!=0 || doubles[i][0][8]!=0
            )   realRows++;
    }

  Tensor res = new  Tensor(realRows,Y,Z);

for(int i = 0; i<realRows; i++){
    for(int j = 0; j<Y; j++){
        for(int k = 0; k<Z; k++){
            res.addElement(i,j,k,doubles[i][j][k]);
        }
    }

}
return res;
}
public void TensorConstDiv(double num){
    int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                tensor[i][j][k] = tensor[i][j][k]/num;
            }
        }
    }
}



public int nonZeroElements(){
    int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    int sum = 0;
    
    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                if(tensor[i][j][k]!=0.0) sum++;
            }
        }
    }
    
    return sum;
}

public Tensor TensorizeSlices(ArrayList<DenseMatrix> MatrixList){
    int X,Y,Z = MatrixList.size();
    Y = MatrixList.get(0).getColumnDimension();
    X = MatrixList.get(0).getRowDimension();
    
    // an den exoun ta idia modalities den ginetai Tensor
    for(int i  = 0; i<MatrixList.size(); i++){
        
        if(X<MatrixList.get(i).getRowDimension()) X = MatrixList.get(i).getRowDimension(); // kratame tis perissoteres grammes
        if(MatrixList.get(i).getColumnDimension()!=Y){
            System.out.println("Matricies have different column Dimensions");
            return null;
        }
    }
    Tensor res = new Tensor(X,Y,Z); 
   // System.out.println("Tensor size: ("+X+","+Y+","+Z+")");
  
    
    // fill tensor elements for Matricies List
   for(int z=0; z<MatrixList.size(); z++){ 
       DenseMatrix Mz = MatrixList.get(z);
   //    System.out.println("Z = ("+Mz.getRowDimension()+","+Mz.getColumnDimension());
        for(int i = 0; i<Mz.getRowDimension(); i++){
            for(int j = 0; j<Mz.getColumnDimension(); j++){
             res.addElement(i, j, z, Mz.getEntry(i, j));
             }
        }
   }
 return res;   
}


public double maxEntry(){
    double res = tensor[0][0][0];
     int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                if(tensor[i][j][k]>res){
                res = tensor[i][j][k];
            }
          }
        }
    }
    return res;
}




public double minEntry(){
    double res = tensor[0][0][0];
     int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                if(tensor[i][j][k]<res){
                res = tensor[i][j][k];
            }
          }
        }
    }
    return res;
}

public Tensor TensorUndersample(double f){
    

    int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    Tensor res = new Tensor(X,Y,Z);
    int prevSize = X*Y*Z;
    int newSize =(int) ((int) prevSize*f);
    int diff = prevSize - newSize;
    int counter = 0;
    
    while(counter<diff){
        
    }
  
        return null;
}

public DenseMatrix getKMatrix(int ith){
    
    int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;
    
    
    DenseMatrix res = new DenseMatrix(X,Y);
    
    for(int i = 0;  i<X; i++){
        for(int j = 0; j<Y; j++){
            res.setEntry(i, j, tensor[i][j][ith]);
        }
    }
    
    return res;
}




 public Tensor MeasurementUndersaple(Tensor Rawdata, double targetFilRatio) {
        double[][][] RawdataTensor = Rawdata.getTensor();

        int X = RawdataTensor.length;
        int Y = RawdataTensor[0].length;
        int Z = RawdataTensor[0][0].length;
        Tensor temp = new Tensor(X,Y,Z);


        int size = X*Y*Z;
        int newSize = (int) (size * targetFilRatio);
        int missMesurements = size - newSize;
        int Missed = 0;
        System.out.println("<>><><><><><><><>F> size = " + size + " targetfilRatio  = " + targetFilRatio + " newSize = " + newSize
                + " " + missMesurements);

        ArrayList<Integer> timestamps = new ArrayList<>();
        ArrayList<Integer> devices = new ArrayList<>();
        ArrayList<Integer> modalities = new ArrayList<>();
        Random randomizer = new Random();
        ArrayList<TensorIndex> indexies = new ArrayList<>();
        int g = 0;
        while(g< missMesurements) {
            int timestamp =  randomizer.nextInt(X );
            int device =  randomizer.nextInt(Y );
            int modality = randomizer.nextInt(Z );


TensorIndex index = new TensorIndex(timestamp,device,modality);

            if(IndexCont(indexies,index)){
               // System.out.println("PERIEXETAI"+timestamp+ ","+device+","+modality);

            }
            else{
                TensorIndex tensorIndex = new TensorIndex(timestamp,device,modality);
                   Unknown.add(tensorIndex);
                  
                Rawdata.setEntry(timestamp,device,modality,0.0);
                System.out.println(Unknown.size()+" MHDENISAME--->"+timestamp+ ","+device+","+modality);
               indexies.add(index);
            Missed++;
                g++;
            }

            for(int iterator = 0; iterator<Unknown.size(); iterator++){
                TensorIndex get = Unknown.get(iterator);
                int x = get.getX(); int y = get.getY(); int z = get.getZ();
                if(UknownMap.get(x) == null){
                    UknownMap.put(x, new HashMap<Integer,Integer>());
                    UknownMap.get(x).put(y, z);
                }
                else{
                     UknownMap.get(x).put(y, z);
                }
                    
            }
        }
         System.out.println("Missed  = " + Missed);

        return Rawdata;
 }

public boolean IndexCont(ArrayList<TensorIndex> list, TensorIndex value){
    for(int i = 0; i<list.size(); i++){
        if(list.get(i).getX()==value.getX()
                &&list.get(i).getY()==value.getY()
                && list.get(i).getZ()==value.getZ()){
            return true;
        }
    }
    return false;
}

public boolean isElementUnknown(int i,int j,int k){
  //  System.out.println("isElement Unknown Callled");
    for(int iter = 0; iter<Unknown.size(); iter++){
        
        TensorIndex get = Unknown.get(iter);
        if(get.getX() == i && get.getY() == j && get.getZ() == k) return true;
        
    }
    // System.out.println("isElement Unknown return");
    return false;
}

public boolean isElementUnknownMap(int i,int j,int k){

        
      if(!UknownMap.containsKey(i)) return false;
      
      else{
          HashMap<Integer, Integer> get = UknownMap.get(i);
          if(!get.containsKey(j)) return false;
          else{
              if(!get.containsValue(k)) return false;
              else return true;
          }
      }
 
}

public double meanEntry(){
    double sum = 0;
    int N=0;
      double res;
     int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;

    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
                sum = tensor[i][j][k];
                N++;
            }
          }
        }
    res = sum/N;
    return res;
}

}
