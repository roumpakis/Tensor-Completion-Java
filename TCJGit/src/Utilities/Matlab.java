package Utilities;

import static TC.Main.matlab;
import tensorcompletion.Tensor;

import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;




import la.decomposition.SingularValueDecomposition;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.DenseVector;
import la.vector.Vector;



/**
 * Created by roubakas on 4/20/2017.
 */

public class Matlab {

    public static double MACHEPS = 2E-16;

    public double[][] randn(int n) {
        double res[][] = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res[i][j] = Math.random();
            }
        }
        return res;
    }

    public double[][] randn(int x, int y) {
        double res[][] = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                res[i][j] = Math.random();
            }
        }
        return res;
    }


    // returns a x by y array with ones elements
    public double[][] ones(int x, int y) {
        double res[][] = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                res[i][j] = 1.0;
            }
        }
        return res;
    }

//    //returns a n by n array with zeros elements
//    public double[][] zeros(int n) {
//        double res[][] = new double[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                res[i][j] = 0.0;
//            }
//        }
//        return res;
//    }

    // returns a x by y array with zeros elements
    public double[][] zeros(int x, int y) {
        double res[][] = new double[x][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < x; j++) {
                res[i][j] = 0.0;
            }
        }
        return res;
    }


    // print a 2D matrix
    public void Matrix2DtoString(double arg[][]) {
        int x = arg.length;
        int y = arg[0].length;
        System.out.println("2D Array [" + x + "," + y + "]");
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                System.out.print(" " + arg[i][j]);
            }
            System.out.println();
        }
    }


    public DenseMatrix ArraytoDeneseMatrix(double[][] arg) {
        int X = arg.length;
        int Y = arg[0].length;

        DenseMatrix res = new DenseMatrix(X, Y);
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                res.setEntry(i, j, arg[i][j]);
            }
        }
        return res;
    }

    public HashMap<Integer, ArrayList<Double>> MeasurementUndersaple(HashMap<Integer, ArrayList<Double>> Rawdata, double targetFilRatio) {
        HashMap<Integer, ArrayList<Double>> res = new HashMap<Integer, ArrayList<Double>>();
        // init data
        for (int i = 0; i < 13; i++) {
            res.put(i, new ArrayList<Double>());
        }

        int size = Rawdata.get(0).size()*9;
        int newSize = (int) (size * targetFilRatio);
        int missMesurements = size - newSize;
        int Missed = 0;
        System.out.println("<>><><><><><><><>F> size = " + Rawdata.get(0).size() + " targetfilRatio  = " + targetFilRatio + " newSize = " + newSize
                + " " + missMesurements);

        ArrayList<Integer> indexs = new ArrayList<Integer>();
        ArrayList<Integer> modalities = new ArrayList<>();
        Random randomizer = new Random();

        int g = 0;
        while(g< missMesurements) {
            int anInt = randomizer.nextInt(Rawdata.get(0).size()) + 0;
            int modality = randomizer.nextInt(11 - 3 + 1) + 3;


                if( Rawdata.get(modality).get(anInt)!=0){
                    Rawdata.get(modality).set(anInt, 0.0);
                  //  System.out.println(anInt+"  ---------->      "+modality);
                    modalities.add(modality);
                    indexs.add(anInt);
                    g++;
                }
                else{
                  //  System.out.println("PERIEXETAI"+anInt+"  XXXX     "+modality);
                }


        }



        System.out.println("Missed  = " + Missed);
//       for(int i = 0; i<indexs.size(); i++){
//           System.out.println(indexs.get(i));
//       }

        return Rawdata;
    }


    public Integer prod(ArrayList<Integer> vector) {
        int res = 1;

        for (int i = 0; i < vector.size(); i++) {
            res = res * vector.get(i);
        }
        return res;
    }

    public Double prod(DenseMatrix vector) {
        Double res = 1.0;

        for (int i = 0; i < vector.getRowDimension(); i++) {
            for (int j = 0; j < vector.getColumnDimension(); j++) {
                res = res * vector.getEntry(i, j);
            }

        }
        return res;
    }

    ArrayList<Integer> ones(int N) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int i = 0; i < N; i++) res.add(i, 1);
        return res;
    }

    ArrayList<Integer> zeros(int N) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int i = 0; i < N; i++) res.add(i, 0);
        return res;
    }

//  public  Double norm(ArrayList<Double> vector){
//        Double sum = 0.0;
//        Double aDouble;
//        Double DSq;
//
//        for(int i =0; i<vector.size(); i++){
//
//            aDouble= vector.get(i);
////            System.out.print(" aDouble  = "+aDouble);
//            if(aDouble<0) aDouble=-aDouble;
//            DSq   = aDouble*aDouble;
////            System.out.print("&&&&&& dsq = "+DSq);
//            sum = sum + DSq;
////            System.out.print(" "+sum+"\n");
//        }
//        System.out.println();
//        double sqrt = Math.sqrt(sum);
//        return sqrt;
//
//    }


    public Double FrobeniusNorm(DenseMatrix matrix) {

        Double sum = 0.0;
        Double SQ;
        double element;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (matrix.getEntry(i, j) < 0) element = -matrix.getEntry(i, j);
                else element = matrix.getEntry(i, j);

                SQ = element * element;
                sum = sum + SQ;

            }
        }
        Double res = Math.sqrt(sum);

        return res;
    }

    public double[][] MatrixConstDivide(double[][] matrix, double con) {
        double[][] res = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                res[i][j] = matrix[i][j] / con;
            }
        }
        return res;
    }

    public void DenseMatrixtoString(DenseMatrix arg) {
        System.out.println("Print Dense Matrix");
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                double p = arg.getEntry(i, j);
                System.out.print(" " + String.format("%.4f", p));
            }
            System.out.println();
        }
    }

    public int sumOne(DenseMatrix arg) {
        int res = 0;
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                if (arg.getEntry(i, j) == 1.0) res++;
            }
        }
        return res;
    }


    public DenseMatrix DenseMatrixMul(DenseMatrix x, DenseMatrix y) {


        int xRowDimension = x.getRowDimension();
        int xColumnDimension = x.getColumnDimension();
        int yRowDimension = y.getRowDimension();
        int yColumnDimension = y.getColumnDimension();


        DenseMatrix res = new DenseMatrix(xRowDimension, yColumnDimension);

        if (x.getColumnDimension() != y.getRowDimension()) {
            System.out.println("Wrong Dimensions!!!!!!!!!");
        } else {

            for (int i = 0; i < xRowDimension; i++) {
                for (int j = 0; j < yColumnDimension; j++) {
                    res.setEntry(i, j, 0.0);
                }
            }

            for (int i = 0; i < xRowDimension; i++) {
                for (int j = 0; j < yColumnDimension; j++) {
                    for (int k = 0; k < xColumnDimension; k++) {
                        double entry = res.getEntry(i, j) + x.getEntry(i, k) * y.getEntry(k, j);
                        res.setEntry(i, j, entry);
                    }
                }
            }
        }
        // System.out.println("[" + xRowDimension + "," + xColumnDimension + "]X[" + yRowDimension + "," + yColumnDimension + "] = "
        //       + res.getRowDimension() + "," + res.getColumnDimension());

        return res;
    }

    //  Mn = Mn(known) - data
    public ArrayList<Double> EstimationKnownDiff(Tensor Mn, ArrayList<Double> data, ArrayList<TensorIndex> known) {
        ArrayList<Double> res = new ArrayList<>();
        for (int i = 0; i < known.size(); i++) {
            TensorIndex tensorIndex = known.get(i);
            int x = tensorIndex.getX();
            int y = tensorIndex.getY();
            int z = tensorIndex.getZ();

            double estimation = Mn.getTensor()[x][y][z];
            double realData = data.get(i);
            double o = estimation - realData;
//
            //  if (i == 0) System.out.println("Est: " + estimation + "-" + realData + " = " + o);
            res.add(i, estimation - realData);

        }

        return res;

    }

    public DenseVector ArrayListtoDenseVector(ArrayList<Double> arg) {
        DenseVector res = new DenseVector(arg.size());
        for (int i = 0; i < arg.size(); i++) {
            res.set(i, arg.get(i));
        }

        return res;
    }

    public double sum(DenseMatrix arg) {
        double res = 0.0;
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                res = res + arg.getEntry(i, j);
            }
        }

        return res;
    }

    public DenseMatrix DenseMatrixConstDiv(DenseMatrix arg, double scale) {
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                arg.setEntry(i, j, arg.getEntry(i, j) / scale);
            }
        }
        return arg;
    }
    public DenseMatrix DenseMatrixConstMul(DenseMatrix arg, double scale) {
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                arg.setEntry(i, j, arg.getEntry(i, j)*scale);
            }
        }
        return arg;
    }

    public DenseMatrix DenseMatrixCopy(DenseMatrix arg) {
        DenseMatrix res = new DenseMatrix(arg.getRowDimension(), arg.getColumnDimension());
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                res.setEntry(i, j, arg.getEntry(i, j));
            }
        }
        return res;
    }


    public DenseMatrix DenseMatrixSub(DenseMatrix arg1, DenseMatrix arg2) {
        DenseMatrix res = new DenseMatrix(arg1.getRowDimension(), arg1.getColumnDimension());
        if (arg1.getRowDimension() == arg2.getRowDimension() && arg1.getColumnDimension() == arg2.getColumnDimension()) {
            for (int i = 0; i < arg1.getRowDimension(); i++) {
                for (int j = 0; j < arg1.getColumnDimension(); j++) {
                    res.setEntry(i, j, arg1.getEntry(i, j) - arg2.getEntry(i, j));
                }
            }
        } else {
            System.out.println("Dimension ERROR!!!!!!!!");
        }
        return res;
    }

    public DenseMatrix pinv(DenseMatrix A) {
        System.out.println("<<<<<<<<<<<<<<<PINVVVVVV>>>>>");
        DenseMatrix res = new DenseMatrix();
//U,S,V
        Matrix[] svd = ml.utils.Matlab.svd(A);
        Matrix U = svd[0];
        Matrix S = svd[1];
        Matrix V = svd[2];

//        SingularValueDecomposition svd = new SingularValueDecomposition(A);
//        Matrix S = svd.getS();
//       Matrix U =  svd.getU();
//        Matrix V = svd.getV();
        DenseMatrix Ut = (DenseMatrix) U.transpose();
        DenseMatrix Si = (DenseMatrix) ml.utils.Matlab.inv(S);

        DenseMatrix mul = (DenseMatrix) ml.utils.Matlab.mtimes(Si,Ut);
        res = (DenseMatrix) ml.utils.Matlab.mtimes(V,mul);
        return res;

    }


    public void DenseMatrixtoFile(DenseMatrix arg) {
        for (int i = 0; i < arg.getRowDimension(); i++) {
            for (int j = 0; j < arg.getColumnDimension(); j++) {
                if (j == arg.getColumnDimension() - 1) System.out.print(arg.getEntry(i, j) + "|||");
                else System.out.print(arg.getEntry(i, j) + ",");
            }
        }
        System.out.println();
    }

    public void DenseMatrixtoString(DenseVector data) {
        for (int i = 0; i < data.getDim(); i++) {
            System.out.println(data.get(i));
        }
    }


    public ArrayList<Double> DenseVector2ArrayList(DenseVector v) {
        ArrayList<Double> res = new ArrayList<>();
        for (int i = 0; i < v.getDim(); i++) {
            res.add(i, v.get(i));
        }
        return res;
    }

    public DenseVector circshift(DenseVector arg, int dim) {


        int n = arg.getDim();
        DenseVector res = new DenseVector(n);
        if (dim == 0) {
            res.set(0, arg.get(0));
            res.set(1, arg.get(1));
            res.set(2, arg.get(2));
        }
        if (dim == 1) {
            res.set(0, arg.get(1));
            res.set(1, arg.get(2));
            res.set(2, arg.get(0));
        }

        if (dim == 2) {
            res.set(0, arg.get(2));
            res.set(1, arg.get(0));
            res.set(2, arg.get(1));
        }

        return res;
    }

    public int[] ArrayListtoIntVector(ArrayList<Integer> arg) {
        int[] res = new int[arg.size()];
        for (int i = 0; i < arg.size(); i++) res[i] = arg.get(i);
        return res;
    }


    public Tensor reshape(DenseMatrix arg, ArrayList<Integer> Nway) {
        Tensor res = new Tensor(Nway.get(0), Nway.get(1), Nway.get(2));
        int resx = 0;
        int resy = 0;
        int resz = 0;


        for (int j = 0; j < arg.getColumnDimension(); j++) {

            for (int i = 0; i < arg.getRowDimension(); i++) {
                res.setEntry(resx, resy, resz, arg.getEntry(i, j));
                resx++;


            }
            resy++;

            resx = 0;
            if (resy == Nway.get(1)) {
                resy = 0;
                resz++;
            }
        }

        return res;
    }


    public Tensor reshape(DenseMatrix arg, DenseVector Nway, int mode) {
        int x = (int) Nway.get(0);
        int y = (int) Nway.get(1);
        int z = (int) Nway.get(2);
       // System.out.println("x = "+x+" y = "+y+"z = "+z);

        int resx = 0;
        int resy = 0;
        int resz = 0;

//        if (mode == 0) {
//            Tensor res = new Tensor(x, y, z);
//            for (int j = 0; j < arg.getColumnDimension(); j++) {
//
//                for (int i = 0; i < arg.getRowDimension(); i++) {
//                    res.addElement(resx, resy, resz, arg.getEntry(i, j));
//                    resx++;
//
//
//                }
//                resy++;
//
//                resx = 0;
//                if (resy == Nway.get(1)) {
//                    resy = 0;
//                    resz++;
//                }
//            }
//        }
//
//
//        if (mode == 1) {
//
//            Tensor res = new Tensor(x, y, z);
//            System.out.println(arg.getRowDimension()+" Q "+arg.getColumnDimension());
//            for (int j = 0; j < arg.getColumnDimension(); j++) {
//
//                for (int i = 0; i < arg.getRowDimension(); i++) {
//                    res.addElement(resx,resy,resz,arg.getEntry(i,j));
//                    System.out.println("resx = "+resx+"resy = "+resy+"resz = "+resz);
//                    resx++;
//                }
//                resx=0;
//                resy++;
//                if(resy==Nway.get(1)){
//                    resz++;
//                    resy=0;
//                }
//            }
//            return res;
//        }
//
//
//
//
//
//
//        if (mode == 2) {
//            Tensor res = new Tensor(x, y, z);
//System.out.println(arg.getRowDimension()+" Q "+arg.getColumnDimension());
//            for (int j = 0; j < arg.getColumnDimension(); j++) {
//
//                for (int i = 0; i < arg.getRowDimension(); i++) {
//                    res.addElement(resx,resy,resz,arg.getEntry(i,j));
//                    System.out.println("resx = "+resx+"resy = "+resy+"resz = "+resz);
//                resx++;
//                }
//                resx=0;
//                resy++;
//                if(resy==Nway.get(1)){
//                    resz++;
//                    resy=0;
//                }
//            }
//
//
//            return res;
//        }



        Tensor res = new Tensor(x, y, z);
        for (int j = 0; j < arg.getColumnDimension(); j++) {

            for (int i = 0; i < arg.getRowDimension(); i++) {
                res.setEntry(resx, resy, resz, arg.getEntry(i, j));
                resx++;


            }
            resy++;

            resx = 0;
            if (resy == Nway.get(1)) {
                resy = 0;
                resz++;
            }
        }



        return res;
    }

    public Tensor Fold(DenseMatrix arg, DenseVector Nway, int mode){
        DenseVector circshiftNway = circshift(Nway, mode);
        Tensor reshape = reshape(arg, circshiftNway, mode);
        Tensor res = shiftdim(reshape,Nway,mode);
        return  res;
    }
    public Tensor shiftdim(Tensor arg,DenseVector Nway,int mode){


        
            int resx = (int) Nway.get(0);
            int resy = (int) Nway.get(1);
            int resz = (int) Nway.get(2);

            Tensor res = new Tensor(resx, resy, resz);

            int argy = arg.getTensor()[0].length;
            int argz = arg.getTensor()[0][0].length;
            int argx = arg.getTensor().length;

            ArrayList<Double> values = new ArrayList<>();
            
              for (int i = 0; i < argx; i++) {
                for (int j = 0; j < argy; j++) {
                    for (int k = 0; k < argz; k++) {
                        values.add(arg.getTensor()[i][j][k]);
                    }
                }
             }
            
          int valuesC = 0;    
if(mode==0){
   
    
    
    
         for (int i = 0; i < argx; i++) {
                for (int j = 0; j < argy; j++) {
                     for (int k = 0; k < argz; k++) {
                        
                        res.setEntry(i, j, k, values.get(valuesC));
                        valuesC++;
                        
                    }
                }
             }
}else if(mode==1){
    for (int i = 0; i < argx; i++) 
                for (int j = 0; j < argy; j++) {
                    for (int k = 0; k < argz; k++) {
         {
              
                   
                res.setEntry(k, i, j, values.get(valuesC));
                        valuesC++;
                        
                    }
                }
             }
}
else{
     
         for (int i = 0; i < argx; i++) {
                for (int j = 0; j < argy; j++) {
                    for (int k = 0; k < argz; k++) {
                    res.setEntry(j, k, i, values.get(valuesC));
                        valuesC++;
                        
                    }
                }
             }

}
System.out.println();
            return res;
    }

    public DenseMatrix Unfold(Tensor arg, DenseVector Nway, int mode) {
        DenseVector circshiftNway = circshift(Nway, mode);

        int resx = (int) circshiftNway.get(0);
        int resy = (int) (circshiftNway.get(1) * circshiftNway.get(2));

        DenseMatrix res = new DenseMatrix(resx, resy);

        resx=0;
        resy=0;



        if(mode==0) {
            for (int i = 0; i < arg.getTensor().length; i++) {
                for (int k = 0; k < arg.getTensor()[0][0].length; k++) {
                    for (int j = 0; j < arg.getTensor()[0].length; j++) {

                        res.setEntry(resx, resy, arg.getTensor()[i][j][k]);
                        //  System.out.println("resx = "+resx+" resy = "+resy+"  "+i+","+j+","+k);
                        resy++;
                    }
                }
                resy = 0;
                resx++;
            }
        }


        if(mode==1) {
            for (int j = 0; j < arg.getTensor()[0].length; j++) {

                for (int i = 0; i < arg.getTensor().length; i++) {
                    for (int k = 0; k < arg.getTensor()[0][0].length; k++) {
                        res.setEntry(resx, resy, arg.getTensor()[i][j][k]);
                        //  System.out.println("resx = "+resx+" resy = "+resy+"  "+i+","+j+","+k);
                        resy++;
                    }
                }
                resy = 0;
                resx++;
            }
        }


        if(mode==2) {
            for (int k = 0; k < arg.getTensor()[0][0].length; k++) {
                for (int j = 0; j < arg.getTensor()[0].length; j++) {
                    for (int i = 0; i < arg.getTensor().length; i++) {


                        res.setEntry(resx, resy, arg.getTensor()[i][j][k]);
                        // System.out.println("resx = "+resx+" resy = "+resy+"  "+i+","+j+","+k);
                        resy++;
                    }
                }
                resy = 0;
                resx++;
            }
        }

        return  res;
    }

    public Jama.Matrix LAML2JAMAMatrix(DenseMatrix arg){
        Jama.Matrix res = new Jama.Matrix(arg.getRowDimension(),arg.getColumnDimension());

        for(int i =0; i<res.getRowDimension(); i++){
            for(int j = 0; j<res.getColumnDimension(); j++){
                res.set(i,j,arg.getEntry(i,j));
            }
        }
        return res;
    }

    public DenseMatrix  JAMA2LAMLMatrix(Jama.Matrix arg){
        DenseMatrix res = new DenseMatrix(arg.getRowDimension(),arg.getColumnDimension());

        for(int i =0; i<res.getRowDimension(); i++){
            for(int j = 0; j<res.getColumnDimension(); j++){
                res.setEntry(i,j,arg.get(i,j));
            }
        }
        return res;
    }



    public  Jama.Matrix Jamapinv( Jama.Matrix x) {
        int rows = x.getRowDimension();
        int cols = x.getColumnDimension();
        if (rows < cols) {
            Jama.Matrix result = Jamapinv(x.transpose());
            if (result != null)
                result = result.transpose();
            return result;
        }
        Jama.SingularValueDecomposition svdX = new Jama.SingularValueDecomposition(x);
        if (svdX.rank() < 1)
            return null;
        double[] singularValues = svdX.getSingularValues();
        double tol = Math.max(rows, cols) * singularValues[0] * MACHEPS;
        double[] singularValueReciprocals = new double[singularValues.length];
        for (int i = 0; i < singularValues.length; i++)
            if (Math.abs(singularValues[i]) >= tol)
                singularValueReciprocals[i] =  1.0 / singularValues[i];
        double[][] u = svdX.getU().getArray();
        double[][] v = svdX.getV().getArray();
        int min = Math.min(cols, u[0].length);
        double[][] inverse = new double[cols][rows];
        for (int i = 0; i < cols; i++)
            for (int j = 0; j < u.length; j++)
                for (int k = 0; k < min; k++)
                    inverse[i][j] += v[i][k] * singularValueReciprocals[k] * u[j][k];
        return new  Jama.Matrix(inverse);
    }



    public static boolean checkEquality( Jama.Matrix A,  Jama.Matrix B) {
        return A.minus(B).normInf() < 1e-9;
    }


    public double sum(DenseVector alpha) {
        double res = 0;
        for(int i = 0; i<alpha.getDim(); i++) res = res+alpha.get(i);
        return res;
    }





    public DenseMatrix Merge(DenseMatrix A, DenseMatrix B){
        if(A.getRowDimension()==B.getRowDimension()) {
            DenseMatrix res = new DenseMatrix(A.getRowDimension(), A.getColumnDimension() + B.getColumnDimension());

            int i ;
            int j;

            for( i =0; i<res.getRowDimension(); i++){
                for(j = 0; j<A.getColumnDimension(); j++){
                   res.setEntry(i,j,A.getEntry(i,j));

                }
                res.setEntry(i,res.getColumnDimension()-1,B.getEntry(i,j-A.getColumnDimension()));
            }




            return res;
        }
        else return null;
    }

public Jama.Matrix luckyRound(Jama.Matrix arg){
    Jama.Matrix res = new Jama.Matrix(arg.getRowDimension(),arg.getColumnDimension());

    for(int i = 0; i<arg.getRowDimension(); i++){
        for(int j = 0; j<arg.getColumnDimension(); j++){
            res.set(i,j,RoundTo2Decimals(arg.get(i,j)));

        }
    }
return res;
}


    double RoundTo2Decimals(double val) {
        DecimalFormat df2 = new DecimalFormat("###########.###########");
        return Double.valueOf(df2.format(val));
    }


public double DenseVectorMin(DenseVector arg){

        double res = arg.get(0);

        for(int i  = 1; i<arg.getDim(); i++){
            if(arg.get(i)<res) res = arg.get(i);
        }
        return res;
    }


    public double DenseVectorMax(DenseVector arg){
        double res = arg.get(0);

        for(int i  = 1; i<arg.getDim(); i++){
            if(arg.get(i)>res) res = arg.get(i);
        }
        return res;
    }



    public double MSE(Tensor original, Tensor reconstructed){
        double res = 0;
        double[][][] originalTensor = original.getTensor();
        double[][][] reconstructedTensor = reconstructed.getTensor();
        int X = originalTensor.length;
        int Y = originalTensor[0].length;
        int Z = originalTensor[0][0].length;
        Tensor temp = new Tensor(X,Y,Z);
        if(originalTensor.length!=reconstructedTensor.length
                ||originalTensor[0].length!=reconstructedTensor[0].length
                ||originalTensor[0][0].length!=reconstructedTensor[0][0].length
                ) res = -1;
        else{
            int N = 0;
            for(int i = 0; i<X; i++){
                for(int j = 0; j<Y; j++){
                    for(int  k = 0; k<Z; k++){
                        double dif = originalTensor[i][j][k] - reconstructedTensor[i][j][k];
                            temp.addElement(i,j,k,dif);
                        N++;
                    }
                }
            }
            DenseVector tensor2ArrayList = Tensor2DenseVector(temp);
            double norm = ml.utils.Matlab.norm(tensor2ArrayList);
            norm = norm*norm;
            res = norm/N;
        }
        return res;
    }
public DenseVector Tensor2DenseVector(Tensor original){

    double[][][] originalTensor = original.getTensor();
    int X = originalTensor.length;
    int Y = originalTensor[0].length;
    int Z = originalTensor[0][0].length;
    DenseVector res = new DenseVector(X*Y*Z);
int count = 0;
    for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int  k = 0; k<Z; k++){

              res.set(count,originalTensor[i][j][k]);
              count++;
            }
        }
    }

    return res;
}


public ArrayList<Double> TMacError (ArrayList<Tensor> original, ArrayList<Tensor> reconstructed){
    ArrayList<Double> res = new ArrayList<Double>();

    for(int i = 0; i<original.size(); i++){
        Tensor Originaltensor = original.get(i);
        double OriginalTensorNorm = ml.utils.Matlab.norm(Tensor2DenseVector(Originaltensor));
Originaltensor.TensorConstDiv(OriginalTensorNorm);

        Tensor Reconstructedtensor = reconstructed.get(i);
        double ReconstructedTensorNorm = ml.utils.Matlab.norm(Tensor2DenseVector(Reconstructedtensor));
        Reconstructedtensor.TensorConstDiv(ReconstructedTensorNorm);

        double mse = MSE(Originaltensor, Reconstructedtensor);
        res.add(mse);
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

                Rawdata.setEntry(timestamp,device,modality,0.0);
                System.out.println("MHDENISAME--->"+timestamp+ ","+device+","+modality);
               indexies.add(index);
            Missed++;
                g++;
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

    /**
     *
     * @param T0
     * @param T1
     * @return
     */
    public ArrayList<Double> TMacElementsError(Tensor T0, Tensor T1){
    ArrayList<Double> res = new ArrayList<Double>();
    
    for(int i = 0; i<T0.getTensor().length; i++){
            for(int j=0; j<T0.getTensor()[i].length; j++){
                for(int k=0; k<T0.getTensor()[i][j].length; k++){
                    
                   res.add(T0.getTensor()[i][j][k] - T1.getTensor()[i][j][k] );
                   //System.out.println("("+i+","+j+","+k+") "+res.get(res.size()-1));
                }
    
    
  
}
}
      return res;
    }
    
    
   public  Tensor TensorUndersampled(Tensor T0, double filRatio){
        Random randomizer = new Random();
        
        int X = T0.getTensor().length;
        int Y = T0.getTensor()[0].length;
        int Z = T0.getTensor()[0][0].length;
        
        int TensorElements = X*Y*Z;
        int newTensorElements = (int) (TensorElements * filRatio);
        int ElementsDiff = TensorElements - newTensorElements;
        
        int Missed = 0;
        
        Tensor res = new Tensor(X,Y,Z);
        
        for(int i = 0; i<X; i++){
            for(int j = 0; j<Y; j++){
                for(int k = 0; k<Z; k++){
                    res.setEntry(i, j, k,T0.getTensor()[i][j][k] ); 
                }
            }
        }
        System.out.println("start = "+TensorElements+" new size "+ newTensorElements + " Diff = "+ ElementsDiff);
        
     while(Missed<ElementsDiff ){
        int missX = randomizer.nextInt(X);
        int missY = randomizer.nextInt(Y);      
        int missZ = randomizer.nextInt(Z);
        
       // boolean crit1 = XHaveElement(missX,missY,missZ,res);
       // boolean crit2 = YHaveElement(missX,missY,missZ,res);
        //boolean crit3 = ZHaveElement(missX,missY,missZ,res);
        
       // if(crit1 && crit2 && crit3){
            
       if(res.getTensor()[missX][missY][missZ]!=0.0){
            res.setEntry(missX, missY, missZ, 0.0);
            Missed++;
            System.out.println("Missed: "+Missed+"("+missX+","+missY+","+missZ+")");
        }
        else{
            System.out.println("NOT 3 Crits");
        }
        
     }
        
        return res;
    }
    
    boolean XHaveElement(int x,int y,int z, Tensor T){
        
        int X = T.getTensor().length;
        int Y = T.getTensor()[0].length;
        int Z = T.getTensor()[0][0].length;
        
        for(int j = 0; j<Y; j++){
            for(int k=0; k<Z; k++){
                if(T.getTensor()[x][j][k]!=0 && j!=y && k!=z){
                    return true;
                }
            }
        }
         //System.out.println("X DEN THA EIXAME: "+ x +","+y+","+z+")");
        return false;
    }



    boolean YHaveElement(int x,int y,int z, Tensor T){
        
        int X = T.getTensor().length;
        int Y = T.getTensor()[0].length;
        int Z = T.getTensor()[0][0].length;
        
        for(int i = 0; i<X; i++){
            for(int k=0; k<Z; k++){
                if(T.getTensor()[i][y][k]!=0 && i!=x && k!=z){
                    return true;
                }
            }
        }
        System.out.println("Y DEN THA EIXAME: "+ x +","+y+","+z+")");
        return false;
    }





    
    boolean ZHaveElement(int x,int y,int z, Tensor T){
        
        int X = T.getTensor().length;
        int Y = T.getTensor()[0].length;
        int Z = T.getTensor()[0][0].length;
        
        for(int i=0; i<X; i++){
            for(int j = 0; j<Y; j++){
                if(T.getTensor()[i][j][z]!=0 && i!=x && j!=y){
                    return true;
                }
            }
        }
         System.out.println("Z DEN THA EIXAME: "+ x +","+y+","+z+")");
        return false;
    }
    
    public double MSE(DenseVector original, DenseVector recon){
        System.out.println(original.getDim());
        if(original.getDim() != recon.getDim()){
            System.out.println("Exoume thema aderfe");
            return -666;
        }
        double sum = 0;
        int N = original.getDim();
        
        for(int i = 0; i<N; i++){
            double diff = original.get(i) - recon.get(i);
            double diffsq = diff*diff;
            sum=sum+diffsq;
        }
        double res = sum/N;
        System.out.println("MSE is: "+res);
        return res;
    }
    
    public Jama.Matrix DenseMatrix2JamaMatrix(DenseMatrix M){
        Jama.Matrix res = new Jama.Matrix(M.getData());
        return res;
    }
     public Jama.Matrix Matrix2JamaMatrix(Matrix M){
         Jama.Matrix res = new Jama.Matrix(M.getData());
         return res;
     }
public DenseMatrix Matrix2DenseMatrix( Matrix M){
    DenseMatrix res = new DenseMatrix(M.getData());
    return res;
}


public DenseMatrix Hankelization(ArrayList<Double> data, int WindowSize, double overlap){
    
    int N = data.size();
    int sameElements = (int) (WindowSize * overlap);
    int newElements = WindowSize - sameElements;
    
    ArrayList<Integer> windowStarts = new ArrayList<Integer>();
    ArrayList<Integer> windowEnds = new ArrayList<Integer>();
     
    
    // brikoume tis arxes kai ta telh twn para99urwn
    int start = 0;
    int end = N;
    int curr = 0;
    System.out.println(start+" ~ "+end);
    while(curr<=end){
        int Wstart = start;
        int Wend = Wstart+WindowSize;
        if(Wend>end) break;
        windowStarts.add(Wstart);
        windowEnds.add(Wend);
        start = start+newElements;
        curr = Wend;
    }
    DenseMatrix res = new DenseMatrix(windowStarts.size(),WindowSize);
    
    // gia ola ta parathura
    for(int i = 0; i<windowStarts.size(); i++){
        int windowStart = windowStarts.get(i);
        int windowEnd = windowEnds.get(i);
          System.out.println(windowStart+" ~ "+windowEnd);
        // gia ola ta stoixeia tou i-th para8urou
        int col = 0;
        for(int j = windowStart; j<windowEnd; j++){
            res.setEntry(i, col, data.get(j));
            System.out.println("("+i+","+col+") = data ->" + j);
            col++;
        }
        
    }
      // ArrayList<Double> Dehankelization = Dehankelization( res, WindowSize, overlap);
    return res;
}


public ArrayList<Double> Dehankelization(DenseMatrix data,int WindowSize,double overlap){
   
   
    int previousN = (int) (WindowSize*overlap);
    
    ArrayList<Double> res = new ArrayList<Double>();
    for(int j = 0; j<data.getColumnDimension(); j++) res.add(data.getEntry(0,j));
    
    for(int i = 1; i<data.getRowDimension(); i++){
        for(int j = previousN; j<data.getColumnDimension(); j++){
            res.add(data.getEntry(i,j));
        }
    }
    
    return res;
}



public Tensor Tensorization (ArrayList<DenseMatrix> data){
    int Z = data.size();
    int X = data.get(0).getRowDimension();
    int Y = data.get(0).getColumnDimension();
    
    Tensor res = new Tensor(X,Y,Z);
            
            for(int z=0; z<Z; z++){
                DenseMatrix M = data.get(z);
                
                for(int  x=0; x<X; x++){
                    for(int y=0; y<Y; y++){
                        res.setEntry(x, y, z, M.getEntry(x, y));
                    }
                }
            }

    return res;
}

public ArrayList<DenseMatrix> HARDeviceData2HankelMatricies (HashMap<Integer,ArrayList<Double>> d1){
    ArrayList<DenseMatrix>  res = new ArrayList<DenseMatrix>();
    
    ArrayList<Double> accx = new ArrayList<Double>();
         ArrayList<Double> accy = new ArrayList<Double>();
         ArrayList<Double> accz = new ArrayList<Double>();
         ArrayList<Double> gyrox = new ArrayList<Double>();
         ArrayList<Double> gyroy = new ArrayList<Double>();
         ArrayList<Double> gyroz = new ArrayList<Double>();
         ArrayList<Double> magx = new ArrayList<Double>();
         ArrayList<Double> magy = new ArrayList<Double>();
         ArrayList<Double> magz = new ArrayList<Double>();
         
         
         for(int i=0; i<d1.size(); i++){
             accx.add(d1.get(1).get(i));
             accy.add(d1.get(2).get(i));
             accz.add(d1.get(3).get(i));
             gyrox.add(d1.get(4).get(i));
             gyroy.add(d1.get(5).get(i));
             gyroz.add(d1.get(6).get(i));
             magx.add(d1.get(7).get(i));
             magy.add(d1.get(8).get(i));
             magz.add(d1.get(9).get(i));
         }
  
        DenseMatrix accxM = matlab.Hankelization(accx,1000,0.5);
        DenseMatrix accyM = matlab.Hankelization(accy,1000,0.5);
        DenseMatrix acczM = matlab.Hankelization(accz,1000,0.5);
        DenseMatrix gyroxM = matlab.Hankelization(gyrox,1000,0.5);
        DenseMatrix gyroyM = matlab.Hankelization(gyroy,1000,0.5);
        DenseMatrix gyrozM = matlab.Hankelization(gyroz,1000,0.5);
        DenseMatrix magxM = matlab.Hankelization(magx,1000,0.5);
        DenseMatrix magyM = matlab.Hankelization(magy,1000,0.5);
        DenseMatrix magzM = matlab.Hankelization(magz,1000,0.5);
        
          ArrayList<DenseMatrix> allM = new ArrayList();
          
          allM.add(accxM);
          allM.add(accyM);
          allM.add(acczM);
          allM.add(gyroxM);
          allM.add(gyroyM);
          allM.add(gyrozM);
          allM.add(magxM);
          allM.add(magyM);
          allM.add(magzM);
          
    
    
    return allM;
}
public ArrayList<Double> dataListUndersample(ArrayList<Double> data, double fillRatio){
    System.out.println("mphka");
    ArrayList<Double> res = new ArrayList<Double>();
    int size = data.size();
    int newSize = (int) ((int) size*fillRatio);
    int elementDiff = size - newSize;
    int counter = 0;
     Random rand = new Random();
System.out.println(elementDiff);
    while(counter<elementDiff){
        int randomNum = rand.nextInt((size-1 - 1) + 1) + 0;
        
        if(data.get(randomNum)!=Double.NaN){
           data.set(randomNum, Double.NaN);
           counter++;
        }
       System.out.println("coutner "+counter);
    }
    
    
    
    return data;
}

public DenseMatrix matlabHankel(ArrayList<Double> data,int windowSize,int step){
    
    int numSamples = data.size();
    int start_point = 0;
    int end_point = start_point+windowSize-1;
    int numWindows = (numSamples-start_point-windowSize+1)/step+1;
    
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
public ArrayList<Double> DenseMatrix2ArrayList(DenseMatrix M){
    ArrayList<Double> res = new ArrayList<>();
    
    for(int i = 0; i<M.getRowDimension(); i++){
        for(int j = 0; j<M.getColumnDimension(); j++){
            res.add(M.getEntry(i,j));
        }
    }
    return res;
}

public boolean hasZeros(ArrayList<Double> in){
    for(int  i = 0; i<in.size();i++){
        if(in.get(i)==0.0) return true;
    }
return false;

}


public boolean hasNaN(ArrayList<Double> in){
    for(int  i = 0; i<in.size();i++){
        if(in.get(i).equals(Double.NaN)) return true;
    }
return false;

}









public DenseMatrix HankelizationR(ArrayList<Double> data, int WindowSize, double overlap){
    
    int N = data.size();
    int sameElements = (int) (WindowSize * overlap);
    int newElements = WindowSize - sameElements;
    
    ArrayList<Integer> windowStarts = new ArrayList<Integer>();
    ArrayList<Integer> windowEnds = new ArrayList<Integer>();
     
    
    // brikoume tis arxes kai ta telh twn para99urwn
    int start = 0;
    int end = N;
    int curr = 0;
    System.out.println(start+" ~ "+end);
    while(curr<=end){
        int Wstart = start;
        int Wend = Wstart+WindowSize;
        if(Wend>end) break;
        windowStarts.add(Wstart);
        windowEnds.add(Wend);
        start = start+newElements;
        curr = Wend;
    }
    DenseMatrix res = new DenseMatrix(WindowSize,windowStarts.size());
    
    // gia ola ta parathura
    for(int i = 0; i<windowStarts.size(); i++){
        int windowStart = windowStarts.get(i);
        int windowEnd = windowEnds.get(i);
          System.out.println(windowStart+" ~ "+windowEnd);
        // gia ola ta stoixeia tou i-th para8urou
        int col = 0;
        for(int j = windowStart; j<windowEnd; j++){
            res.setEntry(col, i, data.get(j));
            System.out.println("("+i+","+col+") = data ->" + j);
            col++;
        }
        
    }
      // ArrayList<Double> Dehankelization = Dehankelization( res, WindowSize, overlap);
    return res;
}


public ArrayList<Double> DehankelizationR(DenseMatrix data,int WindowSize,double overlap){
   ArrayList<Double> res = new ArrayList<Double>();
   
    int previousN = (int) (WindowSize*overlap);
    int row = data.getRowDimension(); 
    int cols = data.getColumnDimension();;
  
    for(int j = 0; j<data.getRowDimension(); j++) res.add(data.getEntry(0,j));
    
    for(int i = 1; i<data.getColumnDimension(); i++){
        for(int j = previousN; j<data.getRowDimension(); j++){
            res.add(data.getEntry(j,i));
        }
    }
    
    return res;
}


}
