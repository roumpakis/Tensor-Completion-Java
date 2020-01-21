package Utilities;



import tensorcompletion.Tensor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import la.matrix.DenseMatrix;
import la.vector.DenseVector;


public class IO {

	public ArrayList<Double> ReadVars( String path,String name) {
//System.out.println("Read Varzz");
		


		ArrayList<Double> measurements = new ArrayList<Double>();

		BufferedReader fileReader = null;
                File file = new File(path+name);
		//System.out.println("Parsing file...");
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			// reading a assert file context.getAssets().open("filename.xml")
			;

			fileReader = new BufferedReader(new FileReader(file));


			//Read the file line by line
			int k = 0;
			while ((line = fileReader.readLine()) != null) {
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
				for (String token : tokens) {
					row.add(Double.parseDouble(token));
				}

				// separate raw data from class, timestamp, and device id


//				datasource.createMeasurement(meas);
				for(int i = 0; i<row.size(); i++){
					measurements.add(row.get(i));
				}


				k++;

//				if (k==10000) break;
			}
                        fileReader.close();
                        
		} catch (Exception e) {
			e.printStackTrace();
		}

		//System.out.println("Populating db " + measurements.size());
		return measurements;
	}
//
//
//
//
//
//
//
//
//
//
//
//
//	public void WriteVars() {
//		System.out.println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
//		String filename = "myfile";
//		String string = "Hello world!";
//		FileOutputStream outputStream;
//
//		try {
//			outputStream = c.openFileOutput(filename);
//			outputStream.write(string.getBytes());
//			outputStream.close();
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//
//
//	}
//
//
//
//	public  void writeToFile(String data) throws IOException {
//		
//		File file = new File(path, "lebentia.txt");
//
//
//		FileOutputStream stream = new FileOutputStream(file);
//		try {
//			stream.write(Integer.parseInt(data));
//		} finally {
//			stream.close();
//		}
//		System.out.println("OLA PHGANE KARA");
//	}
//
//	public static void writeResultsToFile(String filename,Tensor tensor){
//		try {
//
//			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
//			out.println("participant,alg,winsize,featuresNo,dominantFeatures,execTimeFSA,exectimeFE");
////			for (int i=0; i<results.size(); i++){
////				Results res = results.get(i);
////				out.print(res.getParticipantNo()+",");
////				out.print(res.getAlgNo()+",");
////				out.print(res.getWinsizeSec()+",");
////				out.print(res.getFeaturesNo()+",");
////				ArrayList<Integer> domFeats = res.getDominantFeatures();
////				for (int k=0; k<domFeats.size(); k++){
////					out.print(domFeats.get(k)+",");
////				}
////				out.print(res.getExecTimeFS()+",");
////				out.print(res.getExecTimeFE());
////				out.println();
////			}
//			out.print(" STELEIOOOOOOS");
//			out.close();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			System.out.println("LATHOS FIRE");
//			e.printStackTrace();
//		}
//	}
//
//
        
        public ArrayList<Tensor> file2Tensor (String filename, String path, int sensors,int modalities,int period,double overlap){
//           ArrayList<Tensor> res = new ArrayList<Tensor>(); 
           
            String previousLine="";
           ArrayList<ArrayList<Double>> measurements = new ArrayList<ArrayList<Double>>();
            ArrayList<Timestamp> timestampList = new ArrayList<Timestamp>();
		ArrayList<Tensor> res = new ArrayList<Tensor>();
            int step = (int) ((int) period * overlap);
            
            BufferedReader fileReader = null;
                File file = new File(path+filename);
		//System.out.println("Parsing file...");
                
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        

			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                              
                               
                               
                            
                                for(int q=0; q<tokens.length; q++){
                                    if(q!=0) row.add(Double.parseDouble(tokens[q]));
                                    else timestampList.add( Timestamp.valueOf(tokens[0]));
                                }
                              
				measurements.add(row);
                             
                                                   // separate raw data from class, timestamp, and device id
                     Nlines++;

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
               HashMap <Integer, ArrayList<ArrayList<Double>> > data = new HashMap<Integer,ArrayList<ArrayList<Double>>>();
              int time = 0; // time to tensor diastash
               
             for(int u=0; u<measurements.size(); u++){
                 System.out.println("Start: "+timestampList.get(u)+" sensor: "+measurements.get(u).get(0)
                 +" modality: "+ measurements.get(u).get(1)+" "+measurements.get(u).get(2));
            
             
             }
              Timestamp start = timestampList.get(0);
               start.setMinutes(start.getMinutes()-step);
               
              Timestamp end = timestampList.get(timestampList.size()-1);
              
          
              
             Timestamp last = new Timestamp(start.getTime());
              last.setMinutes(last.getMinutes()+period);
             
             
             
              while(!start.after(end)){
                  
                   data.put(time, new ArrayList<>());
                  
                      System.out.println("|W| "+start+" ~ "+last);
                    
                     for(int i=0; i<timestampList.size(); i++){
                         
                        if( (timestampList.get(i).before(last) || timestampList.get(i).equals(last))
                                && (timestampList.get(i).after(start) || timestampList.get(i).equals(start))){
                           data.get(time).add(measurements.get(i));
                             System.out.println("Tensor Element: "+i);
                             
                 }
               
                
               // System.out.println("Start: "+start+" last: "+last+" end: "+end);
              }
                      start= new Timestamp(start.getTime());
                start.setMinutes(start.getMinutes()+period-step);
                
                last=new Timestamp(start.getTime());
                last.setMinutes(last.getMinutes()+period);
                time++;
              
             }
              Tensor Ti = new Tensor(data.size(),sensors,modalities);
              
             System.out.println(data);
             
           return null;
        }
        
        
        public void writeValuesDiffList( ArrayList<DenseVector> valuesList){
            for(int i=0; i<valuesList.size(); i++){
                DenseVector values = valuesList.get(i);
                writeValuesDiff("vd"+i+".csv",values);
            }
            System.out.println("Different Writed!");
        }
        
              public void writeArrayList2File( ArrayList<Double> valuesList,String filename){
           
        try {
System.out.println("YEAHH!!");
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
                        for(int i=0; i<valuesList.size(); i++){
                         
                             out.println(valuesList.get(i));
                        }
			

			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("LATHOS FIRE");
			e.printStackTrace();
		}
            
        }
        
                            public void writeArrayListInteger2File( ArrayList<Integer> valuesList,String filename){
           
        try {

			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
                        for(int i=0; i<valuesList.size(); i++){
                         
                             out.println(valuesList.get(i));
                        }
			

			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("LATHOS FIRE");
			e.printStackTrace();
		}
            
        }
        
        public void writeValuesDiff(String filename,DenseVector values){
            try {

			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
                        for(int i=0; i<values.getDim(); i++){
                         
                             out.println(values.get(i));
                        }
			

			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("LATHOS FIRE");
			e.printStackTrace();
		}
            
        }
        
        public void Tensor2File(Tensor T,String filename){
            double[][][] tensor = T.getTensor();
            int X = tensor.length;
            int Y = tensor[0].length;
            int Z = tensor[0][0].length;
            
            System.out.println("X= "+X+" Y = "+Y+" Z = "+Z);
            String [] val = new String [X*Y*Z];
         int c=0;
         String str = "";
           try {
               
               PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
//              
//               for(int i=0; i<X; i++){
//                   for(int j=0; j<Y; j++){
//                       for(int k=0; k<Z;k++){
//                    
//                         //  System.out.println("i= "+i+" j = "+j+" k = "+k + " value = "+tensor[i][j][k]+" c = "+c);
//                          val[c] = Double.toString(tensor[i][j][k]);
//                          c++;
//                       }
//                   }
//               }
//               System.out.println("C= "+c);
//               for(int v =0; v<val.length; v++){
//                   System.out.println(" v = "+v+ " val: "+val[v]);
//                   if(val[v]==null ){
//                       System.out.println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
//                               + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
//                               + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
//                               + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
//                               + "a");
//                       
//                   }
//                   str = str +"\n"+val[v];
//                 
//               }
                 out.println(T.toString());
                 String s = T.toString();
                String[] split = s.split("\n");
                System.out.println("Split: "+split.length+"\n"+split);
                
           } catch (IOException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }
			//
                      
			

			//out.close();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			System.out.println("LATHOS FIRE");
//			e.printStackTrace();
//		}
        }
        
        public void writeTensor(Tensor T, String filename){
            double[][][] tensor = T.getTensor();
             int X = tensor.length;
    int Y = tensor[0].length;
    int Z = tensor[0][0].length;
    
    
    
    PrintWriter writer;
            try {
                writer = new PrintWriter(filename, "UTF-8");

                for(int i = 0; i<X; i++){
        for(int j = 0; j<Y; j++){
            for(int k = 0; k<Z; k++){
  
writer.println(tensor[i][j][k]);

            }
        }
    }
    writer.close();
            } catch (FileNotFoundException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            } catch (UnsupportedEncodingException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }

    
    


    
        }
        
        public void createRandomTensorFile(String filename,int X,int Y, int Z){
    
             PrintWriter writer;
            try {
                writer = new PrintWriter(filename, "UTF-8");
                for(int i = 0; i<X; i++){
                    for(int j = 0; j<Y; j++){
                        for(int k = 0; k<Z; k++){
                
                                writer.println(i+","+j+","+k+","+Math.random());

            }
        }
    }
    writer.close();
            } catch (FileNotFoundException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            } catch (UnsupportedEncodingException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }

    
            
            
        }
        public DenseMatrix HARModalityMatrix(String filename, String path, double windowSize,double overlap,int select){
             ArrayList<Timestamp> timestamp = new ArrayList<Timestamp>();
	     ArrayList<Double> modalityValues = new ArrayList<Double>();
                int step = (int) ((int) windowSize * overlap);
            
            BufferedReader fileReader = null;
            File file = new File(path+filename);
		//System.out.println("Parsing file...");
                
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        

			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {

				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                                    
                             modalityValues.add(Double.parseDouble(tokens[select]));
                             
                             // Timestamp
                           
                           Timestamp t =  Timestamp.valueOf("2018-01-01 00:00:00");
                           double ti = Double.parseDouble(tokens[10]);
                           t.setSeconds((int) (t.getSeconds()+ti));
                           timestamp.add(t);
                          // System.out.println(t);
                           Nlines++;
                           
			}

                        Timestamp start = timestamp.get(0);
                        Timestamp end = timestamp.get(timestamp.size()-1);
                        Timestamp windowStart = new Timestamp(start.getTime());
                        Timestamp windowEnd = new Timestamp(start.getTime());
                        windowEnd.setMinutes(windowEnd.getMinutes()+(int)windowSize);
                        
                        
                        
                        System.out.println("Start: "+start+" end: "+end);
                        System.out.println("WindowSTart: "+windowStart+" WindowEnd: "+windowEnd);
                        
                        HashMap<Integer,ArrayList<Integer>> iPerWindow = new HashMap<Integer,ArrayList<Integer>>();
                        int winN = 0;
              // end.setMinutes(end.getMinutes()+(int)windowSize);
                        while(windowEnd.before(end)){
                            // do things
                            ArrayList<Integer> windowsI = new ArrayList<Integer>();
                            for(int i = 0; i<timestamp.size(); i++){
                                if( (timestamp.get(i).after(windowStart) ||timestamp.get(i).equals(windowStart))
                                    &&  (timestamp.get(i).before(windowEnd) ||timestamp.get(i).equals(windowEnd)) ){
                                    windowsI.add(i);
                                       if(windowsI.size() == 186) System.out.println("GAMW TO MOUNIIIIIIIIIIIIIIIIIIII "+i);
                                
                            
                                }
                            }
                         
                            iPerWindow.put(winN, windowsI);
                            winN++;
                            
                            // change window
                            windowStart.setMinutes(windowStart.getMinutes()+step);
                            windowEnd.setTime(windowStart.getTime());
                            windowEnd.setMinutes(windowEnd.getMinutes() +(int) windowSize);
                           // System.out.println("WindowSTart: "+windowStart+" WindowEnd: "+windowEnd);
                        }
                         int resRows = iPerWindow.get(0).size();
                         for(int i = 0; i<iPerWindow.size(); i++){
                           if(iPerWindow.get(i).size() > resRows){
                               resRows = iPerWindow.get(i).size(); 
                           }
                         }
                        
                         
                         //System.out.println("sumElements "+sumElements);
                        DenseMatrix array1 = new DenseMatrix(iPerWindow.size(),resRows);
                        int arrayLines = 0;
                        for(int i = 0; i<iPerWindow.size(); i++){
                            ArrayList<Integer> W = iPerWindow.get(i); //185
                            for(int j=0; j<W.size(); j++){

                                
                                array1.setEntry(i, j, modalityValues.get(W.get(j)));
                                // Chang Line
                                arrayLines++;
                        }
                        }
                        //System.out.println("Start: "+start+" Wnindow Start: "+windowStart);
                        //System.out.println("END: "+end+" Wnindow End: "+windowEnd);
                       //System.out.println(timestamp.size()); 
                       IO io = new IO();

                
                 return array1;
                } catch (Exception e) {
			e.printStackTrace();
		}
                // System.out.println(timestamp);
            return null;
        }
        
        public DenseMatrix HAR2Tensor(String filename, String path, double windowSize, double overlap){

                String previousLine="";
                // accel_xyz
                ArrayList<Double> accx = new ArrayList<Double>();
                ArrayList<Double> accy = new ArrayList<Double>();
                ArrayList<Double> accz = new ArrayList<Double>();
                
                // mag_xyz
                ArrayList<Double> magx = new ArrayList<Double>();
                ArrayList<Double> magy = new ArrayList<Double>();
                ArrayList<Double> magz = new ArrayList<Double>();
                
                // gyro_xyz
                ArrayList<Double> gyrox = new ArrayList<Double>();
                ArrayList<Double> gyroy = new ArrayList<Double>();
                ArrayList<Double> gyroz = new ArrayList<Double>();
                
                ArrayList<Timestamp> timestamp = new ArrayList<Timestamp>();
		
                int step = (int) ((int) windowSize * overlap);
            
            BufferedReader fileReader = null;
            File file = new File(path+filename);
		//System.out.println("Parsing file...");
                
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        

			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                                
                                // acce values
                                System.out.println(tokens[1]);
                             accx.add(Double.parseDouble(tokens[1]));
                             accy.add(Double.parseDouble(tokens[2]));
                             accz.add(Double.parseDouble(tokens[3]));
                             
                              // gyro values
                             gyrox.add(Double.parseDouble(tokens[4]));
                             gyroy.add(Double.parseDouble(tokens[5]));
                             gyroz.add(Double.parseDouble(tokens[6]));
                             
                             
                              // mag values
                             magx.add(Double.parseDouble(tokens[7]));
                             magy.add(Double.parseDouble(tokens[8]));
                             magz.add(Double.parseDouble(tokens[9]));
                             
                             // Timestamp
                           
                           Timestamp t =  Timestamp.valueOf("2018-01-01 00:00:00");
                           double ti = Double.parseDouble(tokens[10]);
                           t.setSeconds((int) (t.getSeconds()+ti));
                           timestamp.add(t);
                          // System.out.println(t);
                           Nlines++;
                           
			}

                        Timestamp start = timestamp.get(0);
                        Timestamp end = timestamp.get(timestamp.size()-1);
                        Timestamp windowStart = new Timestamp(start.getTime());
                        Timestamp windowEnd = new Timestamp(start.getTime());
                        windowEnd.setMinutes(windowEnd.getMinutes()+(int)windowSize);
                        
                        
                        
                        System.out.println("Start: "+start+" end: "+end);
                        System.out.println("WindowSTart: "+windowStart+" WindowEnd: "+windowEnd);
                        
                        HashMap<Integer,ArrayList<Integer>> iPerWindow = new HashMap<Integer,ArrayList<Integer>>();
                        int winN = 0;
              // end.setMinutes(end.getMinutes()+(int)windowSize);
                        while(windowEnd.before(end)){
                            // do things
                            ArrayList<Integer> windowsI = new ArrayList<Integer>();
                            for(int i = 0; i<timestamp.size(); i++){
                                if( (timestamp.get(i).after(windowStart) ||timestamp.get(i).equals(windowStart))
                                    &&  (timestamp.get(i).before(windowEnd) ||timestamp.get(i).equals(windowEnd)) ){
                                    windowsI.add(i);
                                }
                            }
                            
                            iPerWindow.put(winN, windowsI);
                            winN++;
                            
                            // change window
                            windowStart.setMinutes(windowStart.getMinutes()+step);
                            windowEnd.setTime(windowStart.getTime());
                            windowEnd.setMinutes(windowEnd.getMinutes() +(int) windowSize);
                           // System.out.println("WindowSTart: "+windowStart+" WindowEnd: "+windowEnd);
                        }
                        int sumElements = 0;
                         for(int i = 0; i<iPerWindow.size(); i++){
                            sumElements = sumElements + iPerWindow.get(i).size();
                         }
                         //System.out.println("sumElements "+sumElements);
                        DenseMatrix array1 = new DenseMatrix(sumElements,9);
                        int arrayLines = 0;
                        for(int i = 0; i<iPerWindow.size(); i++){
                            ArrayList<Integer> W = iPerWindow.get(i);
                            for(int j=0; j<W.size(); j++){
//                                //acc
                                array1.setEntry(arrayLines, 0, accx.get(W.get(j)));
                                array1.setEntry(arrayLines, 1, accy.get(W.get(j)));
                                array1.setEntry(arrayLines, 2, accz.get(W.get(j)));
//                                
//                                //gyro
                                array1.setEntry(arrayLines, 3, gyrox.get(W.get(j)));
                                array1.setEntry(arrayLines, 4, gyroy.get(W.get(j)));
                                array1.setEntry(arrayLines, 5, gyroz.get(W.get(j)));
//                                
//                                //mag
                                array1.setEntry(arrayLines, 6, magx.get(W.get(j)));
                                array1.setEntry(arrayLines, 7, magy.get(W.get(j)));
                                array1.setEntry(arrayLines, 8, magz.get(W.get(j)));
                                
                                
                                // Chang Line
                                arrayLines++;
                        }
                        }
                        //System.out.println("Start: "+start+" Wnindow Start: "+windowStart);
                        //System.out.println("END: "+end+" Wnindow End: "+windowEnd);
                       //System.out.println(timestamp.size()); 
                       IO io = new IO();

                
                 return array1;
                } catch (Exception e) {
			e.printStackTrace();
		}
                // System.out.println(timestamp);
            return null;
        }
        
        public void writeDenseMatrix(String filename,DenseMatrix M){
            PrintWriter writer;
            try {
                writer = new PrintWriter(filename, "UTF-8");
                for(int i = 0; i<M.getRowDimension(); i++){
                    for(int j = 0; j<M.getColumnDimension(); j++){
 
                        writer.print(M.getEntry(i,j));
                        if(j!=0) writer.print(",");
       }
                    writer.println();
    }
    writer.close();
            } catch (FileNotFoundException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            } catch (UnsupportedEncodingException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        
        public DenseMatrix readDenseMatrix(int r,int c,String path,String filename){
  
            DenseMatrix res = new DenseMatrix(r,c);
            
            
               BufferedReader fileReader = null;
            File file = new File(path+filename);

		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        
ArrayList<Double> values = new ArrayList<>();			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                                for(int i=0; i<tokens.length;i++){
                                       values.add(Double.parseDouble(tokens[i]));
                                }
                 
              
                                   Nlines++;
                        }
                        int valuesC=0;
                        for(int ri=0; ri<r; ri++){
                            for(int ci=0; ci<c; ci++){
                                res.setEntry(ri, ci, values.get(valuesC));
                            valuesC++;
                            }
                        }
                     
                   
                    
                    return res;
              
                } catch (Exception e) {
			e.printStackTrace();
		}
                return null;
        }
        
        
        public ArrayList<TensorIndex> readKnown(String path,String filename){
                        
               BufferedReader fileReader = null;
            File file = new File(path+filename);
		//System.out.println("Parsing file...");
                
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        
ArrayList<Double> values = new ArrayList<>();			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                                for(int i=0; i<tokens.length;i++){
                                       values.add(Double.parseDouble(tokens[i]));
                                }
                 
              
                                   Nlines++;
                        }
                     
                   
                    
                
              
                } catch (Exception e) {
			e.printStackTrace();
		}
                return null;
        }
        
        
        public ArrayList<Integer> ReadIntegerList( String path,String name) {


		ArrayList<Integer> measurements = new ArrayList<Integer>();

		BufferedReader fileReader = null;
                File file = new File(path+name);
		final String DELIMITER = ",";
		try {
			String line = "";
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			// reading a assert file context.getAssets().open("filename.xml")
			;

			fileReader = new BufferedReader(new FileReader(file));


			//Read the file line by line
			int k = 0;
			while ((line = fileReader.readLine()) != null) {
				ArrayList<Integer> row = new ArrayList<Integer>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
				for (String token : tokens) {
					row.add(Integer.parseInt(token));
				}

				// separate raw data from class, timestamp, and device id


//				datasource.createMeasurement(meas);
				for(int i = 0; i<row.size(); i++){
					measurements.add(row.get(i));
				}


				k++;

//				if (k==10000) break;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		//System.out.println("Populating db " + measurements.size());
		return measurements;
	}
        
        public Tensor readTensor(int x,int y , int z,String path,String filename){
            Tensor res = new Tensor(x,y,z);
            
            
     
             BufferedReader fileReader = null;
                File file = new File(path+filename);
		//System.out.println("Parsing file...");
                
		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
                       
                        
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			fileReader = new BufferedReader(new FileReader(file));
                           
                        
ArrayList<Double> values = new ArrayList<Double>();
			//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                              
                               
                               
                            
                                for(int q=0; q<tokens.length; q++){
                                    values.add(Double.parseDouble(tokens[q]));
                                }
                              
				
                             
                                                   // separate raw data from class, timestamp, and device id
                     Nlines++;

			}
                        
                        int vc = 0;

    
        
           
                for(int i = 0; i<x; i++){
                    for(int k = 0; k<z; k++){
                        for(int j = 0; j<y; j++){
            res.setEntry(i, j, k, values.get(vc));
            vc++;
        }
    }
}

		} catch (Exception e) {
			e.printStackTrace();
		}
            
            
            

	
                
            return res;
        }
        public void writeTensorIndicies( Tensor t,String filename){
            int X,Y,Z;
            double[][][] tensor = t.getTensor();
            X = tensor.length;
            Y = tensor[0].length;
            Z = tensor[0][0].length;

           HashMap<Integer,TensorIndex> res  = new HashMap<Integer,TensorIndex>();
           
			PrintWriter out;
                        int counter = 1;
            try {
                out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
                for(int k = 0; k<Z; k++){
                for(int j = 0; j<Y; j++){
                    for(int i=0; i<X; i++){
                     	out.write("("+i+","+j+","+k+")"+counter+"\n");
                        counter++;

                    }
                }
            }
                
			out.close();
            } catch (IOException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }
                        
	

   }
           
                public void write2DenseMatrix( DenseMatrix M,String filename){
           int X,Y;
           X = M.getRowDimension();
           Y = M.getColumnDimension();

			PrintWriter out;

            try {
                out = new PrintWriter(new BufferedWriter(new FileWriter(filename, true)));
               for(int i=0; i<X;i++){
                   for(int j=0; j<Y; j++){
                       
                       out.print(M.getEntry(i, j));
                       if(j!=Y-1) out.print(",");
                   }
                   out.print("\n");
               
               }
                
			out.close();
            } catch (IOException ex) {
                Logger.getLogger(IO.class.getName()).log(Level.SEVERE, null, ex);
            }
                        
	

   }
                
                
                 public HashMap<Integer,ArrayList<Double>> ReadHARData( String path,String name) {


	 HashMap<Integer,ArrayList<Double>> res = new  HashMap<Integer,ArrayList<Double>> ();
         
         for(int i = 0; i<12; i++){
             res.put(i,new ArrayList());
         }
                /*
                Column1: Device ID
                Column2: accelerometer x
                Column3: accelerometer y
                Column4: accelerometer z
                Column5: gyroscope x
                Column6: gyroscope y
                Column7: gyroscope z
                Column8: magnetometer x
                Column9: magnetometer y
                Column10: magnetometer z
                Column11: Timestamp
                Column12: Activity Label
                         */
		BufferedReader fileReader = null;
                File file = new File(path+name);
		final String DELIMITER = ",";
		try {
			String line = "";
			//Create the file readerrec(
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();

			// reading a assert file context.getAssets().open("filename.xml")
			;

			fileReader = new BufferedReader(new FileReader(file));


			//Read the file line by line
			int k = 0;
			while ((line = fileReader.readLine()) != null) {
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
				for (String token : tokens) {
					row.add(Double.parseDouble(token));
				}

                                Double device_id =  row.get(0);
                                Double acc_x = row.get(1);
                                Double acc_y = row.get(2);
                                Double  acc_z = row.get(3);
                                Double gyro_x = row.get(4);
                                Double  gyro_y = row.get(5);
                                Double gyro_z = row.get(6);
                               Double mag_x = row.get(7);
                               Double mag_y = row.get(8);
                               Double mag_z = row.get(9);
                               Double time = row.get(10);
                               Double label = row.get(11);
                               
                               res.get(0).add(device_id);
                               res.get(1).add(acc_x);
                               res.get(2).add(acc_y);
                               res.get(3).add(acc_z);
                               res.get(4).add(gyro_x);
                               res.get(5).add(gyro_y);
                               res.get(6).add(gyro_z);
                               res.get(7).add(mag_x);
                               res.get(8).add(mag_y);
                               res.get(9).add(mag_z);
                               res.get(10).add(time);
                               res.get(11).add(label);
				k++;

//				if (k==10000) break;
			}
                        System.out.println("HAR file with "+k+" rows");
		} catch (Exception e) {
			e.printStackTrace();
		}

		//System.out.println("Populating db " + measurements.size());
		return res;
	}
        
                 
                 

                 public int seenInString(String str,char c){
                     int res = 0;
                     
                     
                     for(int i = 0; i<str.length(); i++){
                         if(c==str.charAt(i)) res++;
                     }
                     
                     return res;
                 }
                 
                 public HashMap<Timestamp,Double> readCyprusData(String path,String filename){
                     HashMap<Timestamp,Double> res = new HashMap<Timestamp,Double>();
                     
                     
                     
                   BufferedReader fileReader = null;
            File file = new File(path+filename);

		//Delimiter used in CSV file
		final String DELIMITER = ",";
		try {
			String line = "";
			//Create the file reader
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = dbf.newDocumentBuilder();
			fileReader = new BufferedReader(new FileReader(file));
            		//Read the file line by line
			int Nlines = 0;
			while ((line = fileReader.readLine()) != null) {
                       
				ArrayList<Double> row = new ArrayList<Double>();
				//Get all tokens available in line
				String[] tokens = line.split(DELIMITER);
                              
                                    if(Nlines!=0){
                                        Timestamp time = Timestamp.valueOf(tokens[0]);
                                        Double value = Double.parseDouble(tokens[1]);
                                        res.put(time, value);

                                }
                 
              
                                   Nlines++;
                        }
                     return res;
    
              
                } catch (Exception e) {
			e.printStackTrace();
                        return null;
		}
                
 }
                 

                 
                 
        }

