package Utilities;

/**
 * Created by roubakas on 5/2/2017.
 */

public class TensorIndex {
    private int x;
    private int y;
    private int z;


    public TensorIndex(int x, int y, int z){
        this.x=x;
        this.y=y;
        this.z=z;
    }


//    public TensorIndex(){
//x=-1;
//y=-1;
//z=-1;
//    }

// Setters & Getters
    public void setX(int x) {
        this.x = x;
    }

    public void setY(int y) {
        this.y = y;
    }

    public void setZ(int z) {
        this.z = z;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getZ() {
        return z;
    }

}
