#include <sstream>
#include <string>
#include <iostream>
//#include <opencv2\highgui.h>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\cv.h>
#include "opencv2/opencv.hpp"

// Socket
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define PORT 20236

// Strategie
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "HSV Image";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";


// Pentru testarea detectiei culorilor  //////////////////////////////////////////////
void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//cout << "x,y: " << x << ", " << y;

}
// END Pentru testarea detectiei culorilor /////////////////////////////////////////////


void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}


void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed , bool &gasit)  // MODIFICAT - in gasit punem true daca am gasit obiectul si false daca nu
{
	// Pun aici gasit = false , pentru a nu mai trebui sa il tot pun pe false , inainte de apelul lui trackFilteredObject
	gasit = false;
	
	
	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			
			// Pot comenta partea asta la rularea strategiei ////////////////////////////////////////////
			
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				//cout << x << "," << y;
				drawObject(x, y, cameraFeed);

			}
			
			// END Pot comenta partea asta la rularea strategiei ////////////////////////////////////////


		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);  // Pot comenta partea randul asta la rularea strategiei ///////////////////////
	}
	
	gasit = objectFound;  //  MODIFICAT
}

//H S si V pentru inrange() adversar (ex : roz) 
#define H_MIN_A 171
#define S_MIN_A S_MIN
#define V_MIN_A V_MIN

#define H_MAX_A H_MAX
#define S_MAX_A S_MAX
#define V_MAX_A V_MAX

//H S si V pentru inrange() eu (ex : galben) 
#define H_MIN_E 29
#define S_MIN_E 11
#define V_MIN_E 251

#define H_MAX_E 78
#define S_MAX_E 256
#define V_MAX_E 256

//H S si V pentru inrange() varful meu (ex : verde , de completat)  // CE CULOARE ALEG PENTRU VARFUL MEU ? ???????????????????????????????????????????????????????????????????
#define H_MIN_V 78
#define S_MIN_V 23
#define V_MIN_V 233

#define H_MAX_V 96
#define S_MAX_V 86
#define V_MAX_V 256


#define MAX_PE_LOC 4 // timpul maxim cat pot sta nemiscat ( eventual sa ma rotesc ) , in secunde   -> cat ?????????????????????????????????????

// -> cat ?????????????????????????????????????
#define DURATA_MISCARE_ROTATIE 400 *1000         // cate microsecunde ma rotesc , inainte sa revin la while(1)
#define DURATA_MISCARE_ATAC 1200 *1000           // cate microsecunde ma atac , inainte sa revin la while(1)
#define DURATA_MISCARE_ROTATIE_IMPACT 300 *1000  // cate microsecunde ma rotesc , cand dupa un impact eu si adversarul nu ne miscam
#define DURATA_MISCARE_ATAC_IMPACT 1000 *1000     // cate microsecunde ma duc in fata sau spate , dupa ce m-am rotit in urma unui impact ,dupa care eu si adversarul nu ne miscam
#define DURATA_STOP 100 *1000                    // cate microsecunde ma opresc daca nu gasesc adversarul
// Verificarile pentru rotatie ar trebui sa fie mai dese dacat cele pentru atac , daca dureaza mai putin sa ma rotesc si sa depistez adversarul , decat sa merg spre el si sa il ajung .
// DURATA_MISCARE_ROTATIE_IMPACT probabil este mai mic decat DURATA_MISCARE_ROTATIE , deoarece cand vreau sa ma rotesc unpic , ca sa ma pot misca , rotatia va fi scurta

#define PI 3.1415

// caracterele pe care le trimit la robot
#define FRONT 'f'
#define BACK 'b'
#define RIGHT 'r'
#define LEFT 'l'
#define STOP 's'


//Enumeratie cu actiunile pe care le fac
enum actiune { ROTATIE, ATAC };


// VARIABILE GLOBALE NOI
	
int xa,ya,xa_old,ya_old;  // coordonatele adversarului
int xe,ye,xe_old,ye_old;  // coordonatele mele
int xv,yv,xv_old,yv_old;  // coordonatele varfului meu

int xmax , ymax;          // dimensiunile in pixeli ale imaginii captate de la filmare

double eps = 0.8;     // double , EROAREA pentru operatii matematici -> cat ?????????????????????????????????????????????????????
int eps_impact = 40;  // in PIXELI , EROAREA pentru a verifica daca este impact , deoarece x si y memoreaza centru robotului   -> cat ?????????????????????????????????????????????????????
int eps_pe_loc = 15;  // in PIXELI , EROAREA pentru a verifica daca dupa inpact eu si adversarul stam pe loc -> cat ?????????????????????????????????????????????????????
// Probabil ca eps_impact este mai mare decat eps_pe_loc , deoarece trebuie sa avem limite destul de mari pentru a verifica daca robotii sunt in impact , fiindca x si y memoreaza centru unui robot

// END VARIABILE GLOBALE NOI

// eu , varful meu si adversarul suntem pe aceesai drepta ( coliniari ) ?
// sens_optim_rotatie = l sau r , sensul optim de rotatie
// sens_atac = f sau b , atacam cu fata sau cu spatele
bool coliniare(char &sens_optim_rotatie, char &sens_atac)
{
	int ye2, yv2, ya2;
	double unghi_intre_drepte, teta;
	double xv_tr, yv_tr;  // x si y varf translatat si rotit
	bool col;  // coliniare ?
	
	// trecem in sistem de axe drept
	ye2 = ymax - ye;
	yv2 = ymax - yv;
	ya2 = ymax - ya;
	
	
	unghi_intre_drepte = acos( 1. * fabs(xa - xe) / sqrt(1. * ((xa - xe) * (xa - xe) + (ya2 -ye2) * (ya2 - ye2))) );
	
	if(ya2 > ye2)
		if(xa >= xe)
			teta = unghi_intre_drepte;
		else
			teta = PI - unghi_intre_drepte;
	else
		if(xa >= xe)
			teta = - unghi_intre_drepte;
		else
			teta = -(PI - unghi_intre_drepte);
			
	xv_tr = cos(teta) * (xv - xe) + sin(teta) * (yv2 - ye2);
	yv_tr = - sin(teta) * (xv - xe) + cos(teta) * (yv2 - ye2);
	
	
	// verificam eu , varful meu si adversarul suntem coliniari
	col = (fabs(yv_tr) < eps);  // daca varful este pe axa Ox a sistemului translatat si rotit => coliniare
	
	if(yv_tr >= 0)
		if(xv_tr >= 0)
		{
			// cadranul 1 in sistemul translatat si rotit
			sens_optim_rotatie = RIGHT;
			sens_atac = FRONT;
		}
		else
		{
			// cadranul 2
			sens_optim_rotatie = LEFT;
			sens_atac = BACK;
		}
	else
		if(xv_tr >= 0)
		{
			// cadranul 4
			sens_optim_rotatie = LEFT;
			sens_atac = FRONT;
		}
		else
		{
			// cadranul 3
			sens_optim_rotatie = RIGHT;
			sens_atac = BACK;
		}
	
	return col;
}


// get_sens_mers_intr_o_parte() este asemanatoare cu coliniare() , mereu sens_optim_rotatie va fi cu rotirea astfel incat vom fi cu fata spre ENTRUL RINGULUI , deoarece in main consideram ca sens_atac = 'f'
// calculam doar sens_optim_rotatie , iar in loc de adversar consideram CENTRUL RINGULUI
void get_sens_mers_intr_o_parte(char &sens_optim_rotatie)
{
	int xm = xmax/2 , ym = ymax/2;  // coordonatele centrului ringului
	
	// la fel ca la coliniare() , doar ca cu centrul ringului , in loc de adversar
	int ye2, yv2, ym2;
	double unghi_intre_drepte, teta;
	double xv_tr, yv_tr;  // x si y varf translatat si rotit
	bool col;  // coliniare ?
	
	// trecem in sistem de axe drept
	ye2 = ymax - ye;
	yv2 = ymax - yv;
	ym2 = ymax - ym;
	
	
	unghi_intre_drepte = acos( 1. * fabs(xm - xe) / sqrt(1. * ((xm - xe) * (xm - xe) + (ym2 -ye2) * (ym2 - ye2))) );
	
	if(ym2 > ye2)
		if(xm >= xe)
			teta = unghi_intre_drepte;
		else
			teta = PI - unghi_intre_drepte;
	else
		if(xm >= xe)
			teta = - unghi_intre_drepte;
		else
			teta = -(PI - unghi_intre_drepte);
			
	xv_tr = cos(teta) * (xv - xe) + sin(teta) * (yv2 - ye2);
	yv_tr = - sin(teta) * (xv - xe) + cos(teta) * (yv2 - ye2);
	// END la fel ca la coliniare() , doar ca cu centrul ringului , in loc de adversar
	
	
	// mereu alegem sens_optim_rotatie astfel incat sa ajungem cu varful spre mijlocul ringului
	if(yv_tr >= 0)
		sens_optim_rotatie = RIGHT;
	else
		sens_optim_rotatie = LEFT;
}


bool impact()  // impact() returneaza true daca este impact si false daca nu
{
	return ((abs(xe - xa) < eps_impact) && (abs(ye - ya) < eps_impact));
}


bool stam_pe_loc()  // folosesc functia stam_pe_loc() , daca este impact , pentru a afla daca ma misc in timpul impactului sau daca eu si adversarul impingem si stam pe loc
{
	//verific daca coordonatele mele sunt aceleasi ( sau la fel pot verifica daca coordonatele adversarului sunt aceleasi )
	return ((abs(xe - xe_old) < eps_pe_loc) && (abs(ye - ye_old) < eps_pe_loc));
}


// transmiterea comenzilor la robot prin socket
// return 0 in caz de suces , -1 la eroare
int transmitere_socket(char *com)
{
    printf("%s\n",com);  // afiseaza si pe ecran


    struct sockaddr_in address;
    int sock = 0,i;
    struct sockaddr_in serv_addr;
    char trans[10];

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
  
    memset(&serv_addr, '0', sizeof(serv_addr));
  
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
      
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "193.226.12.217", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
  
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }
  
	// Transmiterea
	
	// Daca stiu ca in com va fo doar un caracter
	send(sock , com , strlen(com) , 0 );
	
	
	// SAU
	//sprintf(trans,"%c",com[0]);
	//send(sock , trans , strlen(trans) , 0 );
	
	/*
    // Daca com are mai multe caractere , mai multe comenzi
    for(i=0;i<strlen(com);i++)
	if(strchr("fbrls",com[i]))
	{
		sprintf(trans,"%c",com[i]);
		send(sock , trans , strlen(trans) , 0 );
		//sleep(1);  // sleep(1) era pentru a astepta intre comenzi
	}
	*/

    // La final se opreste .  // nu se mai opreste
    //sprintf(trans,"s");
    //send(sock , trans , strlen(trans) , 0 );
	
	close(sock);  // inchid socketul - NEAPARAT !

    return 0;
}


int main(int argc, char* argv[])
{
	//some boolean variables for different functionality within this
	//program
	bool trackObjects = true;
	bool useMorphOps = true;
	
	Point p;  // pentru testarea detectiei culorilor 
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x = 0, y = 0;
	//create slider bars for HSV filtering  -  pentru testarea detectiei culorilor 
	//createTrackbars();  // o apelez doar in while(1) pentru detectia culorilor , cand fac detectia culorilor
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.open("rtmp://172.16.254.99/live/nimic");  // de unde vine FILMAREA
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	

	
	// VARIABILE LOCALE IN MAIN NOI
	
	bool gasit;   // boolean care ne spune daca culoarea cautata este in ring , de exemplu : adversarul mai este in ring
	
	actiune ce_fac = ATAC;  // ce facem acum : ROTATIE sau ATAC
	
	time_t timer_rotatie = 0;  // cat timp m-am rotit pana acum , in secunde
	
	char buff[100];  // buff pentru a trimite siruri la transmitere_socket()
	
	char sens_optim_rotatie = RIGHT;  // e mai rapid sa ma rotesc in dreapta sau in stanga
	char sens_atac = FRONT;           // sens_atac = 'f' daca voi ataca cu fata si sens_atac = 'b' faca voi ataca cu spatele
	
	bool am_gasit_impactul = false;  // daca a avut loc un impact
	
	// END VARIABILE LOCALE IN MAIN NOI
	
	
	
	// captam o data imaginea filmata pentru a afla dimensiunile in pixeli ale imaginii : xmax si ymax
	do 
	{
        printf("aflu dimensiunile\n");
		capture.read(cameraFeed);
		waitKey(30);  // delay pentru a citi cum trebuie imaginea
	} while(cameraFeed.empty());
		
	xmax = cameraFeed.size().width;
	ymax = cameraFeed.size().height;

		
	
	// while(1) pentru reglarea si testarea detectiei culurilor  - il comentez cand ruleaza strategia ///////////////////////////////////
	/*
	//create slider bars for HSV filtering  -  pentru testarea detectiei culorilor 
	createTrackbars();
	
	while (1) 
	{
		//store image to matrix
		capture.read(cameraFeed);

  		if(cameraFeed.empty())
		{
			waitKey(30);  // delay pentru a citi cum trebuie imaginea
     		continue;
		}
		
		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		
		// Detectia
		// Reglez H , S si V pana gasesc culoarea
		//gasit = false;  // setez pe false sa vad daca dupa trackFilteredObject() devine true - nu mai e nevoie pentru ca il fac false in trackFilteredObject()
	
		inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed, gasit);
  
		printf("%d\n",gasit);  // vad cat e gasit
		
		//show frames
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);
		setMouseCallback("Original Image", on_mouse, &p);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
     	waitKey(30);  
	}*/
	// END while(1) pentru reglarea si testarea detectiei culurilor  /////////////////////////////////////////////////////////////////////
	
	
	/*
	// Verificarea functiei transmitere_socket()  -  comentez cand ruleaza stategia
	char s_test[10];
	
	sprintf(s_test, "%c", RIGHT);
	transmitere_socket(s_test);
	sleep(2);
	
	sprintf(s_test, "%c", LEFT);
	transmitere_socket(s_test);
	sleep(2);
	
	sprintf(s_test, "%c", FRONT);
	transmitere_socket(s_test);
	sleep(2);
	
	sprintf(s_test, "%c", BACK);
	transmitere_socket(s_test);
	sleep(2);
	
	sprintf(s_test, "%c", STOP);
	transmitere_socket(s_test);
	sleep(2);
	// END Verificarea functiei transmitere_socket()
	*/
	
	
	// STRATEGIA - WHILE(1)
	
	timer_rotatie = time(NULL);  
	
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while(1)
	{
		//store image to matrix
		capture.read(cameraFeed);
   
		if(cameraFeed.empty())
		{
			printf("empty camera\n");
			waitKey(30);  // delay pentru a citi cum trebuie imaginea
			continue;     
		}
		
		printf("Am citit camera\n");
   
		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		
		
		
		// DETECTIA CULORILOR
		// Detectia adversarului
		//gasit = false;  // setez pe false sa vad daca dupa trackFilteredObject() devine true - nu mai e nevoie pentru ca il fac false in trackFilteredObject()
	
		inRange(HSV, Scalar(H_MIN_A, S_MIN_A, V_MIN_A), Scalar(H_MAX_A, S_MAX_A, V_MAX_A), threshold);
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed, gasit);
		
		if(gasit == false)  // daca nu am gasit adversarul , dau comanda STOP pentru DURATA_STOP milisecunde si ma intorc la while(1)
		{
			sprintf(buff, "%c", STOP);
			transmitere_socket(buff);  // buff va fi "s"
			usleep(DURATA_STOP);
			
			waitKey(30);  // delay pentru a citi cum trebuie imaginea
			continue;
		}
		
		// salvem vechile coordonate ale adversarului
		//xa_old = xa;  // nu le folosesc
		//ya_old = ya;
		// xa si ya contin noile coordonate
		xa = x;
		ya = y;
		
		//printf("1 %d %d\n",x,y);
		
		
		// Detectia mea
		//gasit = false;  // setez pe false sa vad daca dupa trackFilteredObject() devine true - nu mai e nevoie pentru ca il fac false in trackFilteredObject()
		
		inRange(HSV, Scalar(H_MIN_E, S_MIN_E, V_MIN_E), Scalar(H_MAX_E, S_MAX_E, V_MAX_E), threshold);
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed, gasit);
		
		if(gasit == false)  // daca nu m-am gasit , ma intorc la while(1)
		{
			waitKey(30);  // delay pentru a citi cum trebuie imaginea
			continue;
		}
		
		// salvem vechile coordonate ale mele
		xe_old = xe;
		ye_old = ye;
		// xe si ye contin noile coordonate
		xe = x;
		ye = y;
		
		//printf("2 %d %d\n",x,y);
		
		
		// Detectia varfului meu
		//gasit = false;  // setez pe false sa vad daca dupa trackFilteredObject() devine true - nu mai e nevoie pentru ca il fac false in trackFilteredObject()
		
		inRange(HSV, Scalar(H_MIN_V, S_MIN_V, V_MIN_V), Scalar(H_MAX_V, S_MAX_V, V_MAX_V), threshold);
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed, gasit);
		
		if(gasit == false)  // daca nu mi-am gasit varful , ma intorc la while(1)
		{
			waitKey(30);  // delay pentru a citi cum trebuie imaginea
			continue;
		}
				
		// salvem vechile coordonate ale varfului meu
		//xv_old = xv;
		//yv_old = yv;
		// xv si yv contin noile coordonate
		xv = x;
		yv = y;
		
		//printf("3 %d %d\n",x,y);
		
		
		
		// STRATEGIA PROPRIU-ZISA
		
		if(impact() == true)  // ne-am ciocnit
		{
			// E IMPACT
			if(am_gasit_impactul  == true)// daca era deja impact , verific daca ambi stam pe loc , daca da atunci ma rotesc si merg in fata sau spate , altfel continuam cu functia coliniare()
			{
				if(stam_pe_loc() == true)
				{
					get_sens_mers_intr_o_parte(sens_optim_rotatie);  // considerm ca sens_atac , calculat la o repetitie a lui while(1) anterioara , este 'f'
					// sens_optim_rotatie va fi spre MIJLOCUL RINGULUI
					
					// daca sena_atac este 'b' , de fapt , atunci schimbam LEFT cu RIGHT si invers  , determinate prin sens_optim_rotatie
					if(sens_atac == BACK)
						if(sens_optim_rotatie == LEFT)
							sens_optim_rotatie = RIGHT;
						else
							sens_optim_rotatie = LEFT;
					
					
					// rotirea pentru a ma debloca din stat pe loc , ciocnit de adversar
					sprintf(buff, "%c", sens_optim_rotatie);
					transmitere_socket(buff);  // buff va fi "r" sau "l"
					usleep(DURATA_MISCARE_ROTATIE_IMPACT);
					
					
					// miscarea in fata sau spate , dupa cum avem salvat in sens_atac
					sprintf(buff, "%c", sens_atac);
					transmitere_socket(buff);  // buff va fi "f" sau "b"   
					usleep(DURATA_MISCARE_ATAC_IMPACT);  // waitKey(DURATA_MISCARE_ATAC);
					
					//waitKey(30);  // nu mai e nevoie pentru ca asteapta de la usleep
					continue;	
				}
			}
			else
				am_gasit_impactul = true;  // acum s-a petrecut impactul	
		}
		else
		{
			// NU E IMPACT 
			am_gasit_impactul = false;
		}
		
			

		if((coliniare(sens_optim_rotatie, sens_atac) == false) && (time(NULL) - timer_rotatie < MAX_PE_LOC)) // daca stau de prea mult pe loc ( si doar ma rotesc ) , fac un atac
		{	
			// ma voi ROTII

			if(ce_fac == ATAC)  // daca inainte am atacat , pornesc un timer ca sa nu ma rotesc mai mult de MAX_PE_LOC seconds , ca sa nu stau prea mult nemiscat
			{			
				timer_rotatie = time(NULL);
				ce_fac = ROTATIE;
			}

			// daca deja ma roteam continui rotatia
			sprintf(buff, "%c", sens_optim_rotatie);  // pun in buff sensul optim de rotatie si ma rotesc
			transmitere_socket(buff);  // buff va fi "r" sau "l"
			usleep(DURATA_MISCARE_ROTATIE);  // waitKey(DURATA_MISCARE_ROTATIE);
			
			//waitKey(30);  // nu mai e nevoie pentru ca asteapta de la usleep
			continue;
		}
		else
		{
			// voi ATACA
			ce_fac = ATAC;

			sprintf(buff, "%c", sens_atac);
			transmitere_socket(buff);  // buff va fi "f" sau "b"
			usleep(DURATA_MISCARE_ATAC);  // waitKey(DURATA_MISCARE_ATAC);
			
			//waitKey(30);  // nu mai e nevoie pentru ca asteapta de la usleep
			continue;			
		}

		waitKey(30);  // in mod normal nu ar trebui sa ajung niciodata aici , ci while(1) sa se reia mai repede
	}
	
	return 0;
}
