
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <unistd.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

#include <vector>
#include <string>


const unsigned int windowWidth = 512, windowHeight = 512;

bool keyboardState[256];

bool clicked = false;




// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) 
{
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) 
	{
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) 
{
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) 
	{
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) 
{
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) 
	{
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}


// row-major matrix 4x4
struct mat4
{
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33)
    {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
    
    mat4 operator*(const mat4& right)
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4
{
    float v[4];
    
    vec4(float x = 0, float y = 0, float z = 0, float w = 1)
    {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    
    vec4 operator*(const mat4& mat)
    {
        vec4 result;
        for (int j = 0; j < 4; j++)
        {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    
    vec4 operator+(const vec4& vec)
    {
        vec4 result(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
        return result;
    }
};

// 2D point in Cartesian coordinates
struct vec2
{
    float x, y;
    
    vec2(float x = 0.0, float y = 0.0) : x(x), y(y) {}
    
    vec2 operator+(const vec2& v)
    {
        return vec2(x + v.x, y + v.y);
    }
    
    
    vec2 operator*(float s)
    {
        return vec2(x * s, y * s);
    }
};

class Camera
{
    vec2 center;
    vec2 halfSize;
    float orientation;
    
public:
    Camera()
    {
        center = vec2(0.0, 0.0);
        halfSize =  vec2(1.0, 1.0);
        orientation = 0;
    }
    
    float getOrientation(){
        return orientation;
    }
    
    float GetCameraOrientation(){
        return orientation;
    }
    
    mat4 GetViewTransformationMatrix()
    {
        mat4 T = mat4(
                      1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      -center.x, -center.y, 0.0, 1.0);
        
        mat4 S = mat4(
                      1.0 / halfSize.x, 0.0, 0.0, 0.0,
                      0.0, 1.0 / halfSize.y, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0);
        
        float g = orientation/180*M_PI;
        
        mat4 R = mat4(cos(g),sin(g),0,0,
                      -sin(g),cos(g),0,0,
                      0,0,1,0,
                      0,0,0,1);
        
        return T * R * S;
    }
    
    mat4 GetInverseTransformationMatrix(){
        mat4 T = mat4(
                      1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      center.x, center.y, 0.0, 1.0);
        
        mat4 S = mat4(
                      halfSize.x, 0.0, 0.0, 0.0,
                      0.0, halfSize.y, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0);
        
        float g = orientation/180 * M_PI;
        
        mat4 R = mat4(cos(g),sin(-g),0,0,
                      sin(g),cos(g),0,0,
                      0,0,1,0,
                      0,0,0,1);
        
        return S * R * T;
    }
    
    
    void SetAspectRatio(int width, int height)
    {
        halfSize = vec2((float)width / height,1.0);
    }
    
    void Quake(float t){
        center = center + vec2((float)0.008*sin(80*t), (float)0.008*cos(70*t+1));
    }
    
    void Move(float dt, float t)
    {
//        if(keyboardState['d']) center = center + vec2(1.0, 0.0) * dt;
//        if(keyboardState['a']) center = center + vec2(-1.0, 0.0) * dt;
//        if(keyboardState['w']) center = center + vec2(0.0, 1.0) * dt;
//        if(keyboardState['s']) center = center + vec2(0.0, -1.0) * dt;
        
        if(keyboardState['a']) if(orientation < 90){orientation = orientation + 40 * dt;}
        if(keyboardState['d']) if(orientation > -90){orientation = orientation - 40 * dt;}
        if(keyboardState['q']) Quake(t);
    }

};

Camera camera;


class Geometry
{
    
protected:
    unsigned int vao;
    
public:
    
    Geometry(){
        glGenVertexArrays(1, &vao);
    }
    
    virtual void Draw() = 0;
};



class Triangle : public Geometry
{
    unsigned int vbo;	// vertex array object id
    
public:
    Triangle()
    {
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(1, &vbo);		// generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { 0, 0, 1, 0, 0, 1 };	// vertex data on the CPU
        
        glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
                     sizeof(vertexCoords),	// size of the vbo in bytes
                     vertexCoords,		// address of the data array on the CPU
                     GL_STATIC_DRAW);	// copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,	// Attribute Array 0
                              2, GL_FLOAT,		// components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);		// stride and offset: it is tightly packed
    }
    
    void Draw() 
    {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLES, 0, 3); // draw a single triangle with vertices defined in vao
    }
    
};


class Quad : public Geometry
{
    unsigned int vbo;	// vertex array object id
    
public:
    Quad()
    {
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(1, &vbo);		// generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { 0, 0, 1, 0, 0, 1, 1, 1 };	// vertex data on the CPU
        
        glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
                     sizeof(vertexCoords),	// size of the vbo in bytes
                     vertexCoords,		// address of the data array on the CPU
                     GL_STATIC_DRAW);	// copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,	// Attribute Array 0
                              2, GL_FLOAT,		// components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);		// stride and offset: it is tightly packed
    }
    
    void Draw()
    {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }
    
};


class Heart: public Geometry
{
    unsigned int vbo;	// vertex array object id
    
public:
    Heart()
    {
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(1, &vbo);		// generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        
        
        static float vertexCoords[180] = {};	// vertex data on the CPU
        
        float angle = 0;
        
        for(int i = 0; i < 30; i++){
            vertexCoords[i*6] = 0.5;
            vertexCoords[i*6+1] = 0.5;
            vertexCoords[i*6+2] = x(angle/180*M_PI);
            vertexCoords[i*6+3] = y(angle/180*M_PI);
            angle = angle + 12;
            vertexCoords[i*6+4] = x(angle/180*M_PI);
            vertexCoords[i*6+5] = y(angle/180*M_PI);
            angle = angle + 12;
        }
        
        glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
                     sizeof(vertexCoords),	// size of the vbo in bytes
                     vertexCoords,		// address of the data array on the CPU
                     GL_STATIC_DRAW);	// copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,	// Attribute Array 0
                              2, GL_FLOAT,		// components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);		// stride and offset: it is tightly packed
    }
    
    float x(float t){
        return pow(sin(t),3)/1.5 + 0.5;
    };
    
    float y(float t){
        return (13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t))/24 + 0.5;
    }
    
    void Draw()
    {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 60);}
    
};

class Circle: public Geometry
{
    unsigned int vbo;	// vertex array object id
    
public:
    Circle()
    {
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(1, &vbo);		// generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        
        
        static float vertexCoords[361] = {};	// vertex data on the CPU
        
        float angle = 0.0;
        
        vertexCoords[0]= 0.5;
        vertexCoords[1]= 0.5;
        
        for(int i = 2; i < 361; i++){
            vertexCoords[i]=x(angle);
            i++;
            vertexCoords[i]=y(angle);
            angle = angle + 1;
            
        }
        
        glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
                     sizeof(vertexCoords),	// size of the vbo in bytes
                     vertexCoords,		// address of the data array on the CPU
                     GL_STATIC_DRAW);	// copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,	// Attribute Array 0
                              2, GL_FLOAT,		// components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);		// stride and offset: it is tightly packed
    }
    
    float x(float t){
        return sin(t)/1.7+0.5;
    };
    
    float y(float t){
        return cos(t)/1.7+0.5;
    }
    
    void Draw()
    {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_FAN, 0, 60);}
    
};

class Star: public Geometry
{
    unsigned int vbo;	// vertex array object id
    
public:
    Star()
    {
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(1, &vbo);		// generate a vertex buffer object
        
        // vertex coordinates: vbo -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        
        
        float angle = M_PI/2;
        float small = M_PI/5;
        
        static float vertexCoords[24] = { 0, 0, x(3,angle), y(3,angle), x(1,angle+small), y(1,angle+small), x(3,angle+2*small), y(3,angle+2*small), x(1,angle+3*small), y(1,angle+3*small),x(3,angle+4*small), y(3,angle+4*small), x(1,angle+5*small), y(1,angle+5*small),x(3,angle+6*small), y(3,angle+6*small),x(1,angle+7*small), y(1,angle+7*small),x(3,angle+8*small), y(3,angle+8*small),x(1,angle+9*small), y(1,angle+9*small), x(3,angle), y(3,angle) };	// vertex data on the CPU
        
        
        
        
        glBufferData(GL_ARRAY_BUFFER,	// copy to the GPU
                     sizeof(vertexCoords),	// size of the vbo in bytes
                     vertexCoords,		// address of the data array on the CPU
                     GL_STATIC_DRAW);	// copy to that part of the memory which is not modified
        
        // map Attribute Array 0 to the currently bound vertex buffer (vbo)
        glEnableVertexAttribArray(0);
        
        // data organization of Attribute Array 0
        glVertexAttribPointer(0,	// Attribute Array 0
                              2, GL_FLOAT,		// components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);		// stride and offset: it is tightly packed
    }
    
    float x(float R, float theta){
        return R*cos(theta);
    }
    
    float y(float R, float theta){
        return R*sin(theta);
    }
    
    void Draw()
    {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_FAN, 0, 12);}
    
};

class Shader
{
    
    
public:
    
    Shader(){}

    
    ~Shader(){}
    
    virtual void CompileShader() = 0;
    
    virtual void UploadColor(vec4& color) = 0;
    
    virtual void UploadM(mat4& M) = 0;
    
    virtual void UploadTime(double t)=0;
    
    virtual void Run() = 0;

};


class heartShader : public Shader{
    
    unsigned int shaderProgram;
    
public:
    heartShader(){
        CompileShader();
    }
    
    ~heartShader(){
        glDeleteProgram(shaderProgram);
    }
    
    void CompileShader(){
        
        const char *vertexSource = R"(
#version 410
        precision highp float;
        
        in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
        uniform vec3 vertexColor;
        uniform mat4 M;
        out vec3 color;
        void main()
        {
            color = vertexColor;
            gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * M;
            // copy position from input to output
        }
        )";
        
        // fragment shader in GLSL
        const char *fragmentSource = R"(
#version 410
        precision highp float;
        uniform double time;
        in vec3 color;			// variable input: interpolated from the vertex colors
        out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
        
        void main()
        {
            fragmentColor = vec4(color[0]*time,0.05,0.1,1); // extend RGB to RGBA
        }
        )";
        
        
        // create vertex shader from string
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
        
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");
        
        // create fragment shader from string
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
        
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");
        
        // attach shaders to a single program
        shaderProgram = glCreateProgram();
        if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
        
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        
        // connect Attrib Array to input variables of the vertex shader
        glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
        
        // connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory
        
        // program packaging
        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
        
    }
    
    
    
    void UploadColor(vec4& color){
        int location = glGetUniformLocation(shaderProgram, "vertexColor");
        if(location>=0) glUniform3fv(location,1,&color.v[0]);
        else printf("uniform vertexColor (heart) cannot be set\n");
    }
    
    void UploadM(mat4& M){
        int location = glGetUniformLocation(shaderProgram, "M");
        if(location>=0) glUniformMatrix4fv(location,1,GL_TRUE,M);
        else printf("uniform M (heart) cannot be set\n");
    }
    
    void UploadTime(double t){
        int location = glGetUniformLocation(shaderProgram, "time");
        if(location>=0) glUniform1d(location,t);
        else printf("uniform time (heart) cannot be set\n");
    }
    
    
    void Run(){
        glUseProgram(shaderProgram);
    }
    
    
};


class normalShader : public Shader{
    
    unsigned int shaderProgram;

public:
    normalShader(){
        CompileShader();
    }
    
    ~normalShader(){
        glDeleteProgram(shaderProgram);
    }
    
    void CompileShader(){
        
        const char *vertexSource = R"(
        #version 410
        precision highp float;
        
        in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
        uniform vec3 vertexColor;
        uniform mat4 M;
        out vec3 color;
        void main()
        {
            color = vertexColor;				 		// set vertex color
            gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * M;
            // copy position from input to output
        }
        )";
        
        // fragment shader in GLSL
        const char *fragmentSource = R"(
        #version 410
        precision highp float;
        
        in vec3 color;			// variable input: interpolated from the vertex colors
        out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
        
        void main()
        {
            fragmentColor = vec4(color, 1); // extend RGB to RGBA
        }
        )";
        
        
        // create vertex shader from string
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
        
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");
        
        // create fragment shader from string
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
        
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");
        
        // attach shaders to a single program
        shaderProgram = glCreateProgram();
        if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
        
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        
        // connect Attrib Array to input variables of the vertex shader
        glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
        
        // connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor"); // fragmentColor goes to the frame buffer memory
        
        // program packaging
        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
    }

    
    void UploadColor(vec4& color){
        int location = glGetUniformLocation(shaderProgram, "vertexColor");
        if(location>=0) glUniform3fv(location,1,&color.v[0]);
        else printf("uniform vertexColor cannot be set\n");
    }
    
    void UploadM(mat4& M){
        int location = glGetUniformLocation(shaderProgram, "M");
        if(location>=0) glUniformMatrix4fv(location,1,GL_TRUE,M);
        else printf("uniform M cannot be set\n");
    }
    
    void UploadTime(double t){}
    
    
    
    void Run(){
        glUseProgram(shaderProgram);
    }
    
 
};



class Material{
    Shader* shader;
    vec4 color;
    
    
public:
    Material(Shader* inputShader, vec4 inputColor){
        shader = inputShader;
        color = inputColor;
    }
    
    void UploadAttributes(){
        shader->UploadColor(color);
    }
    
};



class Mesh{
    Material* material;
    Geometry* geometry;
    
public:
    Mesh(Geometry* inputGeometry, Material* inputMaterial){
        material = inputMaterial;
        geometry = inputGeometry;
    }
    
    void Draw(){
        material->UploadAttributes();
        geometry->Draw();
    }
};


class Object{
    Shader* shader;
    Mesh* mesh;
    vec2 position;
    vec2 scaling;
    float orientation;
    
public:
    Object(Shader* inputShader, Mesh* inputMesh, vec2 inputPosition, vec2 inputScaling, float inputOrientation){
        shader = inputShader;
        mesh = inputMesh;
        position = inputPosition;
        scaling = inputScaling;
        orientation = inputOrientation;
    }
    
    
    void UploadAttributes(){
        mat4 S = mat4(scaling.x,0,0,0,
                      0,scaling.y,0,0,
                      0,0,1,0,
                      0,0,0,1);
        
        float g = orientation/180*M_PI;
        
        mat4 R = mat4(cos(g),sin(g),0,0,
                      -sin(g),cos(g),0,0,
                      0,0,1,0,
                      0,0,0,1);
        
        mat4 T = mat4(1,0,0,0,
                      0,1,0,0,
                      0,0,1,0,
                      position.x,position.y,0,1);
        
        mat4 V = camera.GetViewTransformationMatrix();
        
        mat4 M = R*S*T*V;
        
        shader->UploadM(M);
    }
    
    void SetOrientation(double t){
        orientation = -20*t;
    }
    
    void SpinScaleDown(double s, double angle){
        if(this){ // Make sure this function is not performed on nullptr
            
            orientation = orientation + angle;
            scaling.x = scaling.x * s;
            scaling.y = scaling.y * s;
        }
        
    }
    
    vec2 GetScale(){
        return scaling;
    }
    
    vec2 GetPos(){
        return position;
    }
    
    void Draw(){
        UploadAttributes();
        mesh->Draw();
    }
};



class Scene{
    Shader* shader;
    Shader* heart;

    std::vector<std::vector<Material*>> materials;
    std::vector<std::vector<Geometry*>> geometries;
    std::vector<std::vector<Mesh*>> meshes;
    std::vector<std::vector<Object*>> objects;
    std::vector<std::vector<int>> shapes;
    
    int icoord;
    int jcoord;
    
    
public:
    Scene(){
        shader = 0;
        heart = 0;
    }
    
    void Initialize(){
        shader = new normalShader();
        heart = new heartShader();
        
        
        materials.resize(10, std::vector<Material*>(10, 0));
        geometries.resize(10, std::vector<Geometry*>(10, 0));
        meshes.resize(10, std::vector<Mesh*>(10, 0));
        objects.resize(10, std::vector<Object*>(10, 0));
        shapes.resize(10, std::vector<int>(10, 0));
        
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 10; j++){
                RandomGenerate(i, j);
                
            }
        }
        
    }
    
    ~Scene(){
        materials.clear();
        geometries.clear();
        meshes.clear();
        objects.clear();

        if(shader) delete shader;
        if(heart) delete heart;
    }
    
    void HeartBeat(double t){ //HeartBeat
        heart->Run();
        heart->UploadTime(sin(3*t));
    }
    
    
    void SetOrientation(double t){ //StarSpin
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 10; j++){
                if(shapes[i][j]==3){
                    if(objects[i][j]){
                      objects[i][j]->SetOrientation(t);
                    }
                }
            }
        }
    }
    
    
    void Draw(){ // draw all objects depending on shapes
        for(int row = 0; row < 10; row++){
            for(int col = 0; col < 10; col++){
                
                if(objects[row][col]){
                    if(shapes[row][col]==2){
                        heart->Run();
                        objects[row][col]->Draw();
                        
                    }else{
                        shader->Run();
                        objects[row][col]->Draw();
                    }
                }
                
            }
        }
    }
    
    void Sticky(float x, float y){
        
        if(clicked and icoord!=-1){ //prevent from sticking empty obj to mouse
            MoveObject(icoord, jcoord, x-0.05, y-0.05); // Stick the center to the mouse;
        }
        
    }
    
    void Storeij(int j, int i){
        
        clicked = true;
        
        if (not objects[i][j]) {
            icoord = -1;
            jcoord = -1;
            return;
        }
        
        if(keyboardState['b']){   //Bomb included here
            Destroy(i, j);
            clicked = false;
            return;
        }
        
        icoord = i;
        jcoord = j;
        
        
//        std::string str("stored");
//        printf("%s\n",str.c_str());
//        
//        printf("%d\n",i);
//        printf("%d\n",j);
    }
    
    void Swap(int j, int i){
        
        clicked = false;
        

        if(icoord==-1 and jcoord==-1){   //if click down on an empty obj, do nothing;
            return;
        } else if (j > 9 or i > 9 or j < 0 or i < 0)
        {
            printf("moveback");
            float x = (float) jcoord/5.25-0.91;
            float y = (float) icoord/5.25-0.91;
            MoveObject(icoord, jcoord, x, y);
            return;
        }else if ((not objects[i][j]) or (abs(i-icoord) > 1 or abs(j-jcoord) > 1) )
        {  // if the new position is empty or too far, set the obj to the beginning position (potentially check 3 in a line)
            float x = (float) jcoord/5.25-0.91;
            float y = (float) icoord/5.25-0.91;
            MoveObject(icoord, jcoord, x, y);
            return;
        } else if (not CheckLegal(i, j)){ // if not legal, return to original position
            float x = (float) jcoord/5.25-0.91;
            float y = (float) icoord/5.25-0.91;
            MoveObject(icoord, jcoord, x, y);
            return;
        }else if( (abs(i-icoord) <= 1 and abs(j-jcoord) <= 1) ){ // legal and close enough

            Material* tempMate = materials[icoord][jcoord];
            materials[icoord][jcoord] = materials[i][j];
            materials[i][j] = tempMate;
            
            
            Geometry* tempGeo = geometries[icoord][jcoord];
            geometries[icoord][jcoord] = geometries[i][j];
            geometries[i][j] = tempGeo;
            
            int tempInt = shapes[icoord][jcoord];
            shapes[icoord][jcoord] = shapes[i][j];
            shapes[i][j] = tempInt;
            
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            meshes[icoord][jcoord] = new Mesh(geometries[icoord][jcoord],materials[icoord][jcoord]);
            
            float x = (float) j/5.25-0.91;
            float y = (float) i/5.25-0.91;
            
            
            MoveObject(i, j, x, y);
            
            float xx = (float) jcoord/5.25-0.91;
            float yy = (float) icoord/5.25-0.91;
            
            MoveObject(icoord, jcoord, xx, yy);
            
        }
        
        
    }
    
    bool CheckLegal(int i, int j){
        
        
        std::vector<std::vector<int>> tempShapes = shapes;
        
        int temp = tempShapes[i][j];
        tempShapes[i][j] = tempShapes[icoord][jcoord];
        tempShapes[icoord][jcoord] = temp;
        
        
        for(int i = 1; i < 9; i++){
            if( tempShapes[0][i] != -1 and tempShapes[0][i]==tempShapes[0][i-1] and tempShapes[0][i]==tempShapes[0][i+1] ){
                return true;
            }
            if( tempShapes[i][0] != -1 and tempShapes[i][0]==tempShapes[i-1][0] and tempShapes[i][0]==tempShapes[i+1][0]){
                return true;
            }
            if( tempShapes[9][i] != -1 and tempShapes[9][i]==tempShapes[9][i-1] and tempShapes[9][i]==tempShapes[9][i+1] ){
                return true;
            }
            if( tempShapes[i][9] != -1 and tempShapes[i][9]==tempShapes[i-1][9] and tempShapes[i][9]==tempShapes[i+1][9]){
                return true;
            }
            
            for(int j = 1; j < 9; j++){
                if( tempShapes[i][j] != -1 and tempShapes[i][j]==tempShapes[i-1][j] and tempShapes[i][j]==tempShapes[i+1][j] ){
                    return true;
                }
                if( tempShapes[i][j] != -1 and tempShapes[i][j]==tempShapes[i][j-1] and tempShapes[i][j]==tempShapes[i][j+1]){
                    return true;
                }
            }
        }
        
        
        
        return false;
        
    }
    

    void Three(){
        
        std::vector<vec2> des;
        
        for(int i = 1; i < 9; i++){
            if(objects[0][i] == nullptr or objects[i][0] == nullptr or objects[9][i] == nullptr or objects[i][9] == nullptr){
                // make sure objects do not rotate until all objects are in place
            }
            if( shapes[0][i]==shapes[0][i-1] and shapes[0][i]==shapes[0][i+1] ){
                des.push_back(vec2(0, i));
                des.push_back(vec2(0, i-1));
                des.push_back(vec2(0, i+1));
            }
            if( shapes[i][0]==shapes[i-1][0] and shapes[i][0]==shapes[i+1][0]){
                des.push_back(vec2(i-1, 0));
                des.push_back(vec2(i, 0));
                des.push_back(vec2(i+1, 0));
            }
            if( shapes[9][i]==shapes[9][i-1] and shapes[9][i]==shapes[9][i+1] ){
                des.push_back(vec2(9, i));
                des.push_back(vec2(9, i-1));
                des.push_back(vec2(9, i+1));
            }
            if( shapes[i][9]==shapes[i-1][9] and shapes[i][9]==shapes[i+1][9]){
                des.push_back(vec2(i-1, 9));
                des.push_back(vec2(i, 9));
                des.push_back(vec2(i+1, 9));
            }
            
            for(int j = 1; j < 9; j++){
                if(objects[i][j]==nullptr){
                    return;
                }
                if(shapes[i][j]==shapes[i-1][j] and shapes[i][j]==shapes[i+1][j] ){
                    des.push_back(vec2(i-1, j));
                    des.push_back(vec2(i, j));
                    des.push_back(vec2(i+1, j));
                }
                if(shapes[i][j]==shapes[i][j-1] and shapes[i][j]==shapes[i][j+1]){
                    des.push_back(vec2(i, j-1));
                    des.push_back(vec2(i, j));
                    des.push_back(vec2(i, j+1));
                }
            }
        }
        
        
        for(int i = 0; i < des.size(); i++){
            vec2 coord = des[i];
            int shape = shapes[coord.x][coord.y];
            if(shape == 0 or shape == 1 or shape == 4){
                objects[coord.x][coord.y]->SpinScaleDown(0.9, 20);
            } else if (shape == 2){
                objects[coord.x][coord.y]->SpinScaleDown(0.9, 20);
            }else if (shape == 3){
                objects[coord.x][coord.y]->SpinScaleDown(0.9, 5);
            }
            
            
            
        }
        
        for(int i = 0; i < des.size(); i++){
            vec2 coord = des[i];
            while( (shapes[coord.x][coord.y]==3 and objects[coord.x][coord.y]->GetScale().x>0.004)
                  or (objects[coord.x][coord.y] and objects[coord.x][coord.y]->GetScale().x>0.01) ){
                return;
            }
            
        }
        
        for(int i = 0; i < des.size(); i++){
            
            vec2 coord = des[i];
            Destroy(coord.x,coord.y);
        }
        // make sure to delete all the objects that should be deleted at once.

    }
    
    void RandomDisappear(){
        for(int i = 0 ; i < 10; i++){
            
            for(int j = 0 ; j < 10; j++){
                
                int n = rand() % 1000;
                if(n == 500){
                    Destroy(i, j);
                    
                }
                
            }
        }
    }
    
    
    void MoveObject(int i, int j, float x, float y){  //move [i][j] to (x,y) depending on the shape of the obj
        if(shapes[i][j] == 0 or shapes[i][j] == 1 or shapes[i][j] == 4){
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
        }else if (shapes[i][j] == 2){
            objects[i][j] = new Object(heart, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
        } else if (shapes[i][j] == 3){
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x+0.05,y+0.05), vec2(0.025,0.025),0);
        }
    }


    
    void Destroy(int i, int j){
        
        materials[i][j] = nullptr;
        geometries[i][j] = nullptr;
        shapes[i][j] = -1;
        meshes[i][j] = nullptr;
        objects[i][j] = nullptr;
    }
    
    void Skyfall(){  //depending on camera orientation
        for(int j = 0; j < 10; j++){
            if(camera.GetCameraOrientation()>-45 and camera.GetCameraOrientation()<45){
                if(objects[9][j]==nullptr){
                    RandomGenerate(9, j);
                }
            }else if(camera.GetCameraOrientation()<=-45){
                if(objects[j][0]==nullptr){
                    RandomGenerate(j, 0);
                }
            }else if(camera.GetCameraOrientation()>=45){
                if(objects[j][9]==nullptr){
                    RandomGenerate(j, 9);
                }
            }
            
        }
    }
    
    void RandomGenerate(int i, int j){
        
        int n = rand() % 5;
        
        float x = (float) j/5.25-0.91;
        float y = (float) i/5.25-0.91;
        
        if(n == 0){
            materials[i][j] = new Material(shader, vec4(0,0,1));
            geometries[i][j] = new Triangle();
            shapes[i][j] = 0;
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            
        } else if (n == 1){
            materials[i][j] = new Material(shader, vec4(0,1,0));
            geometries[i][j] = new Quad();
            shapes[i][j] = 1;
            
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            
        } else if (n == 2){
            materials[i][j] = new Material(heart, vec4(1,0,0,0));
            geometries[i][j] = new Heart();
            shapes[i][j] = 2;
            
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            objects[i][j] = new Object(heart, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            
        } else if (n == 3){
            materials[i][j] = new Material(shader, vec4(1,0.9,0));
            geometries[i][j] = new Star();
            shapes[i][j] = 3;
            
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x+0.05,y+0.05), vec2(0.025,0.025),0);
            
        } else if (n == 4){
            
            materials[i][j] = new Material(shader, vec4(0,1,1));
            geometries[i][j] = new Circle();
            shapes[i][j] = 4;
            
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            
        }
    }
    
    void MoveDown(){ // move the above objects down if the obj is null
        if(camera.GetCameraOrientation() > -45 and camera.GetCameraOrientation() < 45){
            for(int i = 0; i < 9; i++){
                for(int j = 0; j < 10; j++){
                    
                    
                    if (objects[i][j]==nullptr){
                        float x = (float) j/5.25-0.91;
                        float y = (float) i/5.25-0.91;
                    
                    
                    
                        materials[i][j] = materials[i+1][j];
                        geometries[i][j] = geometries[i+1][j];
                        shapes[i][j] = shapes[i+1][j];
                        meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
                        if(shapes[i][j] == 0 or shapes[i][j] == 1 or shapes[i][j] == 4){
                            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
                        }else if (shapes[i][j] == 2){
                            objects[i][j] = new Object(heart, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
                        } else if (shapes[i][j] == 3){
                            objects[i][j] = new Object(shader, meshes[i][j], vec2(x+0.05,y+0.05), vec2(0.025,0.025),0);
                        }
                    
                        Destroy(i+1, j);
                        
                    }
                }
            }
        } else if(camera.GetCameraOrientation() <= -45){
            for(int j = 9; j > 0; j--){
                for(int i = 0; i < 10; i++){
                    
                    if (objects[i][j]==nullptr){
                        float x = (float) j/5.25-0.91;
                        float y = (float) i/5.25-0.91;
                        
                        materials[i][j] = materials[i][j-1];
                        geometries[i][j] = geometries[i][j-1];
                        shapes[i][j] = shapes[i][j-1];
                        meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
                        if(shapes[i][j] == 0 or shapes[i][j] == 1 or shapes[i][j] == 4){
                            objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
                        }else if (shapes[i][j] == 2){
                            objects[i][j] = new Object(heart, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
                        } else if (shapes[i][j] == 3){
                            objects[i][j] = new Object(shader, meshes[i][j], vec2(x+0.05,y+0.05), vec2(0.025,0.025),0);
                        }
            
                        Destroy(i, j-1);
                    }
                }
            }
            
        } else if(camera.GetCameraOrientation() >= 45){
            for(int j = 0; j < 9; j++){
                for(int i = 0; i < 10; i++){
                    
                    if (objects[i][j]==nullptr){
                        float x = (float) j/5.25-0.91;
                        float y = (float) i/5.25-0.91;
            
            materials[i][j] = materials[i][j+1];
            geometries[i][j] = geometries[i][j+1];
            shapes[i][j] = shapes[i][j+1];
            meshes[i][j] = new Mesh(geometries[i][j],materials[i][j]);
            if(shapes[i][j] == 0 or shapes[i][j] == 1 or shapes[i][j] == 4){
                objects[i][j] = new Object(shader, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            }else if (shapes[i][j] == 2){
                objects[i][j] = new Object(heart, meshes[i][j], vec2(x,y), vec2(0.1,0.1),0);
            } else if (shapes[i][j] == 3){
                objects[i][j] = new Object(shader, meshes[i][j], vec2(x+0.05,y+0.05), vec2(0.025,0.025),0);
            }
            
            Destroy(i, j+1);}
                }
            }
        }
        
        
        
    }
    


    
};


Scene *scene;

// initialization, create an OpenGL context
void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);
    scene = new Scene();
    scene -> Initialize();
    srand((int) time(0));
    
}

void onExit()
{

	printf("exit");
}


void onIdle(){
    
    // time elapsed since program started, in seconds
    double t = glutGet(GLUT_ELAPSED_TIME) * 0.001;
    // variable to remember last time idle was called
    static double lastTime = 0.0;
    // time difference between calls: time step
    double dt = t - lastTime;
    // store time
    lastTime = t;
    
    camera.Move(dt,t);

    scene->HeartBeat(t);
    
    scene->SetOrientation(t);
    
    scene->Three();
    
    scene->MoveDown();
    
    scene->Skyfall();
    
    if(keyboardState['q']){
        scene->RandomDisappear();
    }
    
    glutPostRedisplay();
}

// window has become invalid: redraw
void onDisplay()
{
    glClearColor(0, 0, 0, 0); // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    
    scene->Draw();
    
    glutSwapBuffers(); // exchange the two buffers
}


void onMouse(int button, int state, int i, int j){
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    
    float x = ((float)i / viewport[2]) * 2.0 - 1.0;
    float y = 1.0 - ((float)j / viewport[3]) * 2.0;
    
    mat4 inv = camera.GetInverseTransformationMatrix();
    
    vec4 p = vec4(x, y, 0, 1) * inv;
    
    int u = (int)floor((p.v[0] + 1.0) * 5.0);
    int v = (int)floor((p.v[1] + 1.0) * 5.0);
    
    if((u < 0) || (u > 9) || (v < 0) || (v > 9)) return;
    
    if(state == GLUT_DOWN) scene->Storeij(u, v);
    if(state == GLUT_UP) scene->Swap(u, v);
    
}


void onMouseMotion(int i, int j){
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    
    float x = ((float)i / viewport[2]) * 2.0 - 1.0;
    float y = 1.0 - ((float)j / viewport[3]) * 2.0;
    
    mat4 inv = camera.GetInverseTransformationMatrix();
    
    vec4 p = vec4(x, y, 0, 1) * inv;
    
    x = p.v[0];
    y = p.v[1];
    
    scene->Sticky(x,y);
    
}


void onKeyboard(unsigned char key, int x, int y)
{
    keyboardState[key] = true;
}

void onKeyboardUp(unsigned char key, int x, int y)
{
    keyboardState[key] = false;
}

void onReshape(int winWidth0, int winHeight0)
{
    camera.SetAspectRatio(winWidth0, winHeight0);
    glViewport(0, 0, winWidth0, winHeight0);
}



int main(int argc, char * argv[]) 
{
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight); 	// application window is initially of resolution 512x512
	glutInitWindowPosition(50, 50);			// relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Triangle Rendering");

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	
	onInitialization();

	glutDisplayFunc(onDisplay); // register event handlers
    
    glutIdleFunc(onIdle);
    
    glutMouseFunc(onMouse);
    
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    
    glutReshapeFunc(onReshape);
    
    glutMotionFunc(onMouseMotion);

	glutMainLoop();
    
	onExit();
    
	return 1;
}

