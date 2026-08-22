#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <raylib.h>
using namespace std;



//next time, matricies should be defined in terms of the layers.... just the layer before and the layer after... word....




random_device rd;
mt19937 gen1(rd());
mt19937 gen2(rd());
uniform_real_distribution<float> distrib1(-1, 1);
uniform_real_distribution<float> distrib2(-1, 1);

float learning_rate = 0.2;
int input_nodes = 3; 
int hidden_nodes = 12;
int output_nodes = 1;

float actual_val;



class matrix {
    public:
        vector<int> size;
        vector<vector<float>> values;
    //constructors
    matrix(int width, int height){
        size = vector<int>(2, 0);
        size[0] = width;
        size[1] = height;
        values = vector<vector<float>>(width, vector<float>(height, 0));
        int a,b;
        for(a=0; a<width;a++ ){
            for(b=0; b<height; b++){
                values[a][b] = distrib1(gen1);
            }
        }
    }
    vector<float> front_multiply(vector<float> vector_in){
        int i,j;
        vector<float> vector_out = vector<float>(size[1], 0);
        for(j=0; j<size[1]; j++){
            for(i=0; i<size[0]; i++ ){
                vector_out[j] += values[i][j] * vector_in[i]; 
            }
        }
        return vector_out; 
    }
    vector<float> transpose_multiply(vector<float> vector_in){ 
        int i,j;
        vector<float> vector_out = vector<float>(size[0], 0);
        for(i=0; i<size[0]; i++){
            for(j=0; j<size[1]; j++ ){
                vector_out[i] += values[i][j] * vector_in[j]; 
            }
        }
        return vector_out;
    }
};


class layer {
    public:
        int size;
        vector<float> activation;
        vector<float> error;
        vector<float> sigma;
    //constructor
    layer(int s){
        size = s;
        activation = vector<float>(s, 0);
        error = vector<float>(s, 0);
        sigma = vector<float>(s, 0);
    }
    void compute_error(matrix M,vector<float> sigma_prev){ //previous is actually the layer after...called prev because this func is used in back prop.
        error = M.transpose_multiply(sigma_prev);
    }
    void compute_sigma(){ 
        int a;
        for(a=0; a < activation.size(); a++ ){
            sigma[a] = error[a] * (1 - activation[a]*activation[a])/2; ////based on derivitive of the sigmoid function .
        }
    }
};



vector<float> sigmoid(vector<float> vector_in) {
    int b;
    for(b=0; b < vector_in.size(); b++){
        vector_in[b] = (2.0f / (1.0f + exp(-vector_in[b])))-1.0f;
    }
    return vector_in;
}




void update_matrix(matrix& M, layer prev, layer current){ //backward pass frame           activation from prev in foward direction and sigma from backward pass...
    int i,j;
    for(i=0; i< M.size[0]; i++){
        for(j=0; j< M.size[1]; j++){
            M.values[i][j] += learning_rate * current.activation[i] * prev.sigma[j]; 
        }
    }
}



int main(){
    layer input(input_nodes);
    layer hidden(hidden_nodes);
    layer output(output_nodes);
    matrix M1(input_nodes, hidden_nodes);
    matrix M2(hidden_nodes, output_nodes);


    int a,b1,b2;

    InitWindow(600, 600, "ai finding a circle");
    Image img = GenImageColor(60, 60, BLACK);
    Texture2D texture = LoadTextureFromImage(img);

    SetTextureFilter(texture, TEXTURE_FILTER_POINT);

    Color* pixels = (Color*)img.data; //
    //SetTargetFPS(60); //
    while (!WindowShouldClose()){
        learning_rate = ((learning_rate-0.05)*0.99) + 0.05;
        float time = GetTime();
        int index =0;
        for(b1=0; b1<60; b1++){
            for(b2=0; b2<60; b2++){

                input.activation[0] = -1 + (b1*0.033333);
                input.activation[1] = -1 + (b2*0.033333);
                input.activation[2] = 1;

                //predict value
                hidden.activation = sigmoid(M1.front_multiply(input.activation));
                output.activation = sigmoid(M2.front_multiply(hidden.activation));

                unsigned char r = (unsigned char)((output.activation[0] + 1.0f) * 127.5f);
                unsigned char g = (unsigned char)((output.activation[0] + 1.0f) * 127.5f);
                unsigned char b = (unsigned char)((output.activation[0] + 1.0f) * 127.5f);

                pixels[index] = (Color){ r, g, b, 255 };
                index += 1;
            

            }
        }

        UpdateTexture(texture, img.data);

        BeginDrawing();
            ClearBackground(BLACK);
            Rectangle src = { 0.0f, 0.0f, 60.0f, 60.0f };
            Rectangle dest = { 0.0f, 0.0f, 600.0f, 600.0f }; // Scaled up to screen width/height
            Vector2 origin = { 0.0f, 0.0f };
            
            DrawTexturePro(texture, src, dest, origin, 0.0f, WHITE);
            DrawCircleLines(300, 300, 212.0f, RED);
            for(a=0; a< hidden_nodes; a++){
                float z = 0.1;
                float y11 = ((((-M1.values[0][a])+(z)+M1.values[2][a])/M1.values[1][a])+1)*300;
                float y12 = ((((M1.values[0][a])+(z)+M1.values[2][a])/M1.values[1][a])+1)*300;
                float y21 = ((((-M1.values[0][a])+M1.values[2][a]+(z)-(0.01*M2.values[a][0]))/M1.values[1][a])+1)*300;
                float y22 = ((((M1.values[0][a])+M1.values[2][a]+(z)-(0.01*M2.values[a][0]))/M1.values[1][a])+1)*300;

                float intensity = 255/((1+exp(-M2.values[a][0]))*(1+exp(-M2.values[a][0])));

                cout << y11 << endl;
                cout << y12 << endl;
                cout << y21 << endl;
                cout << y22 << endl;
                cout << " " << endl;
                cout << M1.values[1][a] << endl;
                cout << " " << endl;
                DrawLine(0, y11, 600, y12, {0,255,0,static_cast<unsigned char>(intensity)});
                DrawLine(0, y21, 600, y22, {255,0,0,static_cast<unsigned char>(intensity)});
                //{static_cast<unsigned char>(140-intensity), static_cast<unsigned char>(125+intensity),0,static_cast<unsigned char>(125+intensity)}
            }

        EndDrawing();


        input.activation[0] = distrib2(gen2);
        input.activation[1] = distrib2(gen2);
        input.activation[2] = 1;


        // find actual val
        if((input.activation[0]*input.activation[0]) +  (input.activation[1]*input.activation[1]) < 0.5){
            actual_val = 1;
        }
        else {
            actual_val = -1;
        }

        //predict value
        hidden.activation = sigmoid(M1.front_multiply(input.activation));
        output.activation = sigmoid(M2.front_multiply(hidden.activation));

        //back prop
        output.error[0] =   actual_val - output.activation[0];
        output.compute_sigma();

        hidden.compute_error(M2, output.sigma);
        hidden.compute_sigma();

        update_matrix(M2, output, hidden);
        update_matrix(M1, hidden, input);
    }
    UnloadImage(img);
    UnloadTexture(texture);
    CloseWindow();
    return 1;
}