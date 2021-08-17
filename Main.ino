#include <Arduino.h>
#include <math.h>
#include <Servo.h>

#define DEBUG
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif
#define trigPin 6
#define echoPin 7

Servo servoFront, servoLeft, servoRight;

int PH1, PH2, PH3, PH4;
int PHEN = 38;
int menu = 1;
int curr_menu = 1;
int prog_start = 0;
int menu_items = 3;
int distance;

#define APHASE 4
#define APWM 3
#define BPHASE 5
#define BPWM 6
#define left_b 7
#define right_b 2

#define LEDYEL 25
#define LEDRED 26

// Network Configuration - customized per network

const int PatternCount = 8;
const int InputNodes = 3;
const int HiddenNodes = 4;
const int OutputNodes = 2;
const float LearningRate = 0.3;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0015;




float Input[PatternCount][InputNodes] = {
  { 0, 0, 0},  // no obstacle
  { 0, 0, 1},  // obstacle on the right
  { 0, 1, 0},  // obstacle on the front
  { 0, 1, 1},  // obstacle on the front and right
  { 1, 0, 0},  // obstacle on the left
  { 1, 0, 1},  // obstacle on the left and right
  { 1, 1, 0},  // obstacle on the left and front
  { 1, 1, 1},  // obstacle on all sides
};

const float Target[PatternCount][OutputNodes] = {
  { 0.75, 0.75 },   // forward full speed
  { 0.60, 0.75 },   // right motor faster
  { 0.2, 0.4 },     // backward, left faster
  { .5, 0.75 },     // left motor stopped, right fast
  { 0.75, 0.60 },   // left motor faster
  { 0.65, 0.65 },   // forward slow speed
  { 0.75, 0.5 },    // Left motor faster, right stopped
  { 0.5, 0.5},      //Motors stopped
};

// End of Network Configuration

int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error = 2;
float Accum;

float Hidden[HiddenNodes];
float Output[OutputNodes];

float HiddenWeights[InputNodes + 1][HiddenNodes];
float OutputWeights[HiddenNodes + 1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes + 1][OutputNodes];

int ErrorGraph[64];




void setup() {
    for (int x = 0; x < 64; x++) {
    ErrorGraph[x] = 47;
  }

  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(left_b, INPUT_PULLUP);
  pinMode(right_b, INPUT_PULLUP);

  servoLeft.attach(9);
  servoRight.attach(10);
  servoFront.attach(11);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(PHEN, OUTPUT);
  digitalWrite(PHEN, LOW);




  randomSeed(analogRead(A1));       
  ReportEvery1000 = 1;
  for ( p = 0 ; p < PatternCount ; p++ ) {
    RandomizedIndex[p] = p ;
  }

  Serial.begin(115200);
  delay(100);


  int testmode = 1;
  testmode = digitalRead(left_b);

}

void loop() {

  train_nn();

  delay(5000);
  drive_nn();


}


//Servo 1
void motorA(int percent) {
  int maxSpeed = 90;
  int minSpeed = 10;
  int dir = 0;
  if (percent < 50) {
    dir = 0;
  }
  if (percent > 50) {
    dir = 1;
  }
  if (dir == 1) {
    pinMode(APHASE, INPUT);
    pinMode(APWM, INPUT);
    pinMode(APHASE, OUTPUT);
    pinMode(APWM, OUTPUT);
    digitalWrite(APHASE, LOW);
    int drive = map(percent, 51, 100, 0, 255);
    drive = constrain(drive, 0, 1023);

    analogWrite(APWM, drive);
  }
  if (dir == 0) {
    pinMode(APHASE, INPUT);
    pinMode(APWM, INPUT);
    pinMode(APHASE, OUTPUT);
    pinMode(APWM, OUTPUT);
    digitalWrite(APHASE, HIGH);
    int drive = map(percent, 49, 0, 0, 255);
    drive = constrain(drive, 0, 1023);
    Serial.print("Driving Back: ");
    Serial.println(drive);
    analogWrite(APWM, drive);
  }
  if (percent == 50) {
    pinMode(APHASE, INPUT);
    pinMode(APWM, INPUT);
  }
}


//Servo 2
void motorB(int percent) {
  int maxSpeed = 85;
  int minSpeed = 45;
  int dir = 0;
  if (percent < 50) {
    dir = 0;
  }
  if (percent > 50) {
    dir = 1;
  }
  if (dir == 1) {
    pinMode(BPHASE, INPUT);
    pinMode(BPWM, INPUT);
    pinMode(BPHASE, OUTPUT);
    pinMode(BPWM, OUTPUT);
    digitalWrite(BPHASE, HIGH);
    int drive = map(percent, 51, 100, 0, 255);
    drive = constrain(drive, 0, 1023);
    Serial.print("Driving Foreward: ");
    Serial.println(drive);
    analogWrite(BPWM, drive);
  }
  if (dir == 0) {
    pinMode(BPHASE, INPUT);
    pinMode(BPWM, INPUT);
    pinMode(BPHASE, OUTPUT);
    pinMode(BPWM, OUTPUT);
    digitalWrite(BPHASE, LOW);
    int drive = map(percent, 49, 0, 0, 255);
    drive = constrain(drive, 0, 1023);
    analogWrite(BPWM, drive);
  }
  if (percent == 50) {
    pinMode(BPHASE, INPUT);
    pinMode(BPWM, INPUT);
  }
}




// Training the neural network
// ---------------------------------------------------------------------------------------------
void train_nn() {
  // Initialize HiddenWeights and ChangeHiddenWeights
  prog_start = 0;
  for ( i = 0 ; i < HiddenNodes ; i++ ) {
    for ( j = 0 ; j <= InputNodes ; j++ ) {
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100)) / 100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  

  // -------------------------------------------------------------------------------------------------
  // Initialize OutputWeights and ChangeOutputWeights

  for ( i = 0 ; i < OutputNodes ; i ++ ) {
    for ( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;
      Rando = float(random(100)) / 100;
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  //Serial.println("Initial/Untrained Outputs: ");
  //toTerminal();







  //-----------------------------------------------------------------
  //Begin training


  for ( TrainingCycle = 1; TrainingCycle < 2147483647; TrainingCycle++) {


    //Randomize order of training patterns

    for ( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ;
      RandomizedIndex[p] = RandomizedIndex[q];
      RandomizedIndex[q] = r;
    }

    Error = 0.0;


    //Cycle through each training pattern in the randomized order

    for ( q = 0 ; q < PatternCount ; q++ ) {
      p = RandomizedIndex[q];


      //Compute hidden layer activations

      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        Accum = HiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));


      }


      //Compute output layer activations and calculate errors
      for ( i = 0 ; i < OutputNodes ; i++ ) {
        Accum = OutputWeights[HiddenNodes][i] ;
        for ( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum)) ;
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }
      //Serial.println(Output[0]*100);




      //Backpropagate errors to hidden layer

      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        Accum = 0.0 ;
        for ( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }




      // Update Inner-->Hidden Weights
      // =====================================================================================================================

      for ( i = 0 ; i < HiddenNodes ; i++ ) {
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for ( j = 0 ; j < InputNodes ; j++ ) {
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      // Update Hidden-->Output Weights
      // ---------------------------------------------------------------------------------------------------------------------

      for ( i = 0 ; i < OutputNodes ; i ++ ) {
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for ( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    // Print after every 100 epoch
    // --------------------------------------------------------------------------------------------
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {


      // Epoch, Error, Hidden Weight 0 - 15, Output Weight 0 - 7


      Serial.print((float)Error, 5);
      Serial.print(",");
      for ( i = 0 ; i < InputNodes ; i ++ )
      {
        for (j = 0; j < HiddenNodes; j++)
        {

          Serial.print((float)HiddenWeights[j][i]);
          Serial.print(",");


        }

      }

      for ( i = 0 ; i < OutputNodes ; i ++ )
      {
        for (j = 0; j < HiddenNodes; j++)
        {
          Serial.print((float)OutputWeights[j][i]);
          Serial.print(",");

        }

      }
      Serial.print((int)TrainingCycle);
      Serial.println();
      delay(1000);


      /*
               Serial.print ("Epoch: ");
               Serial.print (TrainingCycle);
               Serial.print ("\t");
               Serial.print ("Error = ");
               Serial.println (Error, 5);


               Serial.println();
               Serial.println("Hidden Weights:");
               for ( i = 0 ; i < HiddenNodes ; i ++ )
               {
                 for (j = 0; j < HiddenNodes; j++)
                 {
                   Serial.println(HiddenWeights[j][i]);
                 }

               }

               Serial.println();
               Serial.println("Output Weights:");
               for ( i = 0 ; i < OutputNodes ; i ++ )
               {
                 for (j = 0; j < HiddenNodes; j++)
                 {
                   Serial.println(OutputWeights[j][i]);
                 }

               }

      */
      //toTerminal();

      if (TrainingCycle == 1)
      {
        ReportEvery1000 = 99;
      }
      else
      {
        ReportEvery1000 = 25;
      }
    }




    if ( Error < Success ) break ; // if error rate is less than pre-determined threshold
  }
}






// run trained Neural Network
// --------------------------------------------------------------------------------------

void drive_nn()
{
  Serial.println();
  Serial.println("Trained Neural Network Test");
  
  if (Success < Error) {
    prog_start = 0;
    Serial.println("NN not Trained");
  }
  while (Error < Success) {
    int num;
    int farDist = 20;
    int closeDist = 4;
    float TestInput[] = {0, 0, 0, 0};
    int USS1, USS2, USS3;
    /*
        USS1 = analogRead(A1);   // Collect sonar distances.
        USS2 = analogRead(A2);   // Collect sonar distances.
        USS3 = analogRead(A3);   // Collect sonar distances.
        USS4 = analogRead(A4);   // Collect sonar distances.
    */

    for (int i = 0; i <= 180; i++) {
      delay(15);
      servoFront.write(i);
      if (i == 45) {
        USS1 = ultrasonicSensor();
        //Serial.print(USS1);
      }
      else if (i == 90) {
        USS2 = ultrasonicSensor();
        //Serial.print(USS2);
      }
      else if (i == 135) {
        USS3 = ultrasonicSensor();
        //Serial.print(USS3);
      }
    }
    servoFront.write(0);
    delay(600);



#ifdef DEBUG
    Serial.println();
    Serial.print("Distance: ");
    //Serial.print(USS1);
    Serial.print("\t");
    //Serial.print(USS2);
    Serial.print("\t");
    //Serial.print(USS3);
    Serial.println();
    //Serial.println(USS4);
#endif


    USS1 = map(USS1, 3, 20, 0, 100);
    USS2 = map(USS2, 3, 20, 0, 100);
    USS3 = map(USS3, 3, 20, 0, 100);
    //USS4 = map(USS4, 3, 20, 0, 100);

    USS1 = constrain(USS1, 0, 100);
    USS2 = constrain(USS2, 0, 100);
    USS3 = constrain(USS3, 0, 100);
    //USS4 = constrain(USS4, 0, 100);

    TestInput[0] = float(USS1) / 100;
    TestInput[1] = float(USS2) / 100;
    TestInput[2] = float(USS3) / 100;
    //TestInput[3] = float(USS4) / 100;

#ifdef DEBUG
    Serial.println();
    Serial.print("Input: ");
    //Serial.print(TestInput[3], 2);
    //Serial.print("\t");
    Serial.print(TestInput[2], 2);
    Serial.print("\t");
    Serial.print(TestInput[1], 2);
    Serial.print("\t");
    Serial.println(TestInput[0], 2);
#endif

    InputToOutput(TestInput[0], TestInput[1], TestInput[2], TestInput[3]); //INPUT to ANN to obtain OUTPUT

    int speedA = Output[0] * 100;
    int speedB = Output[1] * 100;
    speedA = int(speedA);
    speedB = int(speedB);
#ifdef DEBUG
    Serial.print("Speed: ");
    Serial.print(speedA);
    Serial.print("\t");
    Serial.println(speedB);
#endif
    motorA(speedA);
    motorB(speedB);
    delay(50);
  }
}




// reading from the ultrasonic sound sensor

int ultrasonicSensor() {
  delay(5);
  long duration;
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 20000);
  distance = duration * 0.034 / 2;
  if (distance > 20 || distance == 0) distance = 20;
  return distance;
}






// Training information
// -----------------------------------------------------------------------------------------------

void toTerminal()
{

  for ( p = 0 ; p < PatternCount ; p++ ) {
    Serial.println();
    Serial.print ("  Training Pattern: ");
    Serial.println (p);
    Serial.print ("  Input ");
    for ( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for ( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }

    // -----------------------------------------------------------------------------------------------


    // Compute hidden layer activations
    // -----------------------------------------------------------------------------------------------
    for ( i = 0 ; i < HiddenNodes ; i++ ) {
      Accum = HiddenWeights[InputNodes][i] ;
      for ( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;


    }

    // -----------------------------------------------------------------------------------------------
    // Compute output layer activations and calculate errors
    for ( i = 0 ; i < OutputNodes ; i++ ) {
      Accum = OutputWeights[HiddenNodes][i] ;
      for ( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0 / (1.0 + exp(-Accum)) ;

    }

  }
}

void InputToOutput(float In1, float In2, float In3, float In4)
{
  float TestInput[] = {0, 0, 0, 0};
  TestInput[0] = In1;
  TestInput[1] = In2;
  TestInput[2] = In3;
  TestInput[3] = In4;




  //Compute hidden layer activations
  // ---------------------------------------------------------------------------------------------

  for ( i = 0 ; i < HiddenNodes ; i++ ) {
    Accum = HiddenWeights[InputNodes][i] ;
    for ( j = 0 ; j < InputNodes ; j++ ) {
      Accum += TestInput[j] * HiddenWeights[j][i] ;
    }
    Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;

    //Serial.print("Hidden Weight: ");
    //Serial.print(Hidden[i]);
  }

  // Compute output layer activations and calculate errors
  // ---------------------------------------------------------------------------------------------

  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Accum = OutputWeights[HiddenNodes][i] ;
    for ( j = 0 ; j < HiddenNodes ; j++ ) {
      Accum += Hidden[j] * OutputWeights[j][i] ;
    }
    Output[i] = 1.0 / (1.0 + exp(-Accum)) ;


    //Serial.print("Output Weight: ");
    //Serial.print(Output[i]);
  }

}
/*
  #ifdef DEBUG

  Serial.print ("Output:  ");
  for ( i = 0 ; i < OutputNodes ; i++ ) {
    Serial.print (Output[i], 5);
    Serial.print (" ");
  }
  #endif
  }

*/
