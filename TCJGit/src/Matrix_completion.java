import static java.lang.Math.*;
import static java.lang.System.out;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import ml.recovery.MatrixCompletion;
import static ml.utils.Printer.*;
import static ml.utils.Time.*;

public class Matrix_completion {
    
    Timestamp start;
    Timestamp end;
    int secs;
    int intervsNo;
    int sensorsNo;
    double[][] initial_data; 
    double[][] reconstructed_data; 
    la.matrix.Matrix data_mtx;
    la.matrix.Matrix REC_DATA;
     public Matrix_completion(){
         System.out.println("Matrix Completion!");
     }
    public Matrix_completion(String formDate, String formStart_time, String formEnd_time, String formFoo,
                String Sid1, String Sid2, String Sid3, String Sid4, String Sid5, String Sid6, 
                String Sid7, String Sid8, String Sid9, String Sid10) throws InstantiationException, IllegalAccessException{
        
        //fix this constructor, we want measurements from these specific sensors given from form
        
        start = Timestamp.valueOf(formDate+" "+formStart_time+":00");
        end = Timestamp.valueOf(formDate+" "+formEnd_time+":00");
        secs = Integer.parseInt(formFoo);
        
        // Number of rows of measurements table
        sensorsNo = 10;
        // Number of columns of measurements table
        intervsNo = findTableSlot(end, start, secs);
        
        initial_data = new double[sensorsNo][intervsNo];
        reconstructed_data = new double[sensorsNo][intervsNo];
        
        out.print("Start timestamp: "+start+" secs");
        out.print("End timestamp: "+end+" secs");
        out.print("Time interval duration : "+secs+" secs");
        out.print("Number of time intervals : "+intervsNo);
    }
    
    public int connect()  throws InstantiationException, IllegalAccessException {
            String url = "jdbc:mysql://localhost:3306/";
            String dbName = "acciona_hbnplatform_v0";
            String driver = "com.mysql.jdbc.Driver";
            String userName = "root";
            String password = "";
            
            try {
                Class.forName(driver).newInstance();
            }   
            catch (ClassNotFoundException e) {
                out.println( e.getMessage( ) );
            } 

            try {
                Connection con = DriverManager.getConnection(url + dbName, userName, password);
                out.println("Connected to database\n");
                Statement stmt = con.createStatement( );
                String query = "SELECT * FROM measurements WHERE timestamp BETWEEN '"+start +"' AND '"+end+"'";
                ResultSet rs = stmt.executeQuery(query);
                 
                if(rs.next()){
                    while (rs.next()) {
                        int id = rs.getInt("id");
                        Timestamp timestamp = rs.getTimestamp("timestamp");

                        int slot = findTableSlot(timestamp, start, secs);
                        out.println("Slot "+slot);

                        String sensor_tag = rs.getString("sensor_tag");
                        int sensorNo = 0;
                        if (sensor_tag.length() == 6) {
                            sensorNo = Integer.parseInt(sensor_tag.substring(5, 6));
                            out.println("SensorNo "+sensorNo);
                        }
                        else if (sensor_tag.length() == 7){
                            sensorNo = Integer.parseInt(sensor_tag.substring(5, 7));
                            out.println("SensorNo "+sensorNo);
                        }
                        double value = rs.getDouble("value");

                        // Indexing begins from 0
                        initial_data[sensorNo-1][slot-1] = value;

                        out.println(id+" "+timestamp+" "+sensor_tag+" "+value);  
                    }
                   matr_compl(initial_data);
                }
                else {
                    return -1;
                }
            }
            catch ( SQLException err ) {
                out.println( err.getMessage( ) );
            }
            return 1;
    }
    
     private static int findTableSlot(Timestamp curr, Timestamp start, int secs){      
        
        String currTime = curr.toString().substring(11, 19);
        out.println("\nCurr time "+currTime);
        int currSec = toSecs(currTime);
        
        String startTime = start.toString().substring(11, 19);
        out.println("Start time "+startTime);
        int startSec = toSecs(startTime);
                
        double slot = 1;
        if(startSec != currSec)
            slot = ceil((double)(currSec-startSec)/secs);
        
        return (int)slot;
        
     }
     
     private static int toSecs(String s) {
        
        String[] hourMin = s.split(":");
        int hour = Integer.parseInt(hourMin[0]);
        int mins = Integer.parseInt(hourMin[1]);
        int secs = Integer.parseInt(hourMin[2]);
        int hoursInSecs = (hour * 60 * 60) +  (mins * 60) + secs;
        return hoursInSecs;
        
    }
     
     public void matr_compl(double[][] data) {
                
        data_mtx = new la.matrix.DenseMatrix(data);
        out.println("\nData Measurements");
        //printMatrix(data_mtx);
        println(data_mtx.toString());
        
        int numCols = data_mtx.getColumnDimension();
        int numRows = data_mtx.getRowDimension();
//        out.println(numRows);
//        out.println(numCols);
        
        int i, j;
        double[][] omega;
        omega = new double [numRows][numCols]; 
        
        for(i=0; i < numRows; i++) {
            for(j=0; j<numCols; j++){
                if(data[i][j]!=0) {
                    omega[i][j] = 1;
                }
                else{
                    omega[i][j] = 0;
                }
            } 
        }    
        
        la.matrix.Matrix omega_mtx;
        omega_mtx = new la.matrix.DenseMatrix(omega);
    
        out.println("Indices");
        //printMatrix(omega_mtx);
        println(omega_mtx.toString());
                
        out.println("INITIALIZING MATRIX COMPLETION ...\n");
        tic();
        
        // Run matrix completion
        MatrixCompletion matrixCompletion = new MatrixCompletion();
        matrixCompletion.feedData(data_mtx);
        matrixCompletion.feedIndices(omega_mtx);
        matrixCompletion.run();
       
        // OutputmatrixCompletion
        REC_DATA = matrixCompletion.GetLowRankEstimation();
        out.println("Reconstructed Matrix");
        //printMatrix(REC_DATA.transpose());
        println(REC_DATA.transpose().toString());
        
        out.println(" ... ENDING MATRIX COMPLETION\n");
   
        fprintf("Elapsed time for matrix completion: %.2f seconds.%n", toc());
        
        setRecData();

    }
    
    private void setRecData(){        

        int numRows = REC_DATA.getRowDimension();
        int numCols = REC_DATA.getColumnDimension();
        
        //out.println(numRows);
        //out.println(numCols);
        
        int i, j;
        for(j=0; j < numCols; j++){
            for(i=0; i < numRows; i++) {
               reconstructed_data[j][i] = REC_DATA.getEntry(i, j);
               out.println(reconstructed_data[j][i]);
            } 
        } 
        
         /*for(i=0; i < numRows; i++) {
            for(j=0; j < numCols; j++){
                if(reconstructed_data[j][i] == 0) { 
                 out.print("wrong");
                }
               //out.println(reconstructed_data[j][i]);
            } 
        } */
        //out.print(reconstructed_data[0][1]);
        //out.print(reconstructed_data[2][1]);
        //out.print(reconstructed_data[9][0]);

    }
    
    public double[][] getInitData(){
        return initial_data;
    }
    
    public double[][] getRecData(){
        return reconstructed_data;
    }
    
    public la.matrix.Matrix getInitDataMatrix(){
        return data_mtx;
    }
    
    public la.matrix.Matrix getRecDataMatrix(){
        return REC_DATA.transpose();
    }
    
   
}
