program enigmamachine;
// (c) 2012 dangermouse
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp;

const
  DEBUG = false;
  NONE  = -1;
  BETA  = 9;
  GAMMA = 10;

  IOC_TRESHOLD = 1.45;

type
  TAlphabet = Array[1..26] of Char;
  PAlphabet = ^TAlphabet;

type
  TEnigmaApplication = class(TCustomApplication)
  protected
    procedure DoRun; override;
    procedure DoTest;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;

  private
    ciphertext_  : AnsiString;
    rotor_,
    invrotor_    : Array[1..10] of TAlphabet;  // I..VII rotors plus Beta and Gamma rotors of Naval Enigma
    rotturnpos_  : Array[1..8] of Array[1..2] of Longint;     // Beta and Gamma rotors do not rotate, rotors VI, VII, VIII have two notches
    plugboard_   : TAlphabet;
    reflector_   : Array[1..4] of TAlphabet;   // B, C, B thin, C thin

    freq_        : Array[1..26] of Longint;
    bestioc_     : Extended;

    procedure readAlphabet(a : AnsiString; p : PAlphabet);
    procedure invertRotor(rot : Longint);
    procedure initEnigma;
    function  isValidRotors(rot1, rot2, rot3, rot4 : Longint) : Boolean;
    function  cipher(c : AnsiString; pos1, pos2, pos3, pos4, rot1, rot2, rot3, rot4, refl : Longint) : AnsiString;
    function  cipherChar(c : AnsiString; i, pos1, pos2, pos3, pos4, rot1, rot2, rot3, rot4, refl : Longint) : Char;
    procedure nextPosition(var pos1, pos2, pos3 : Longint; rot1, rot2, rot3 : Longint);
    function shiftPos(ch : Char; posrot: Longint) : Longint;

    function calculateIndexOfCoincidence(plain : AnsiString) : Extended;
    procedure outputValidPlainText(plain : AnsiString);
    procedure breakArmyEnigma(c : AnsiString);
    procedure breakNavalEnigma(c : AnsiString);
end;

procedure TEnigmaApplication.readAlphabet(a : AnsiString; p : PAlphabet);
var i : Longint;
begin
 if (length(a)<>26) then raise Exception.Create('Alphabet is not of length 26!');
 for i:=1 to 26 do
    p^[i] := a[i];
end;

procedure TEnigmaApplication.invertRotor(rot : Longint);
var i : Longint;
begin
  for i:=1 to 26 do
    begin
      // normally i -> rotor_[rot][i]
      // inverted rotor_[rot][i] -> i;
      invrotor_[rot][Ord(rotor_[rot][i])-64] := Chr(i+64);
    end;
end;

procedure TEnigmaApplication.initEnigma;
var F : TextFile;
    j,i : Longint;
    s : AnsiString;
begin
 AssignFile(F, 'solutions.txt');
 Rewrite(F);
 CloseFile(F);


 // where the turnover happens for each rotor
 rotturnpos_[1][1] := Ord('Q')-64;
 rotturnpos_[2][1] := Ord('E')-64;
 rotturnpos_[3][1] := Ord('V')-64;
 rotturnpos_[4][1] := Ord('J')-64;
 rotturnpos_[5][1] := Ord('Z')-64;
 rotturnpos_[6][1] := Ord('Z')-64;
 rotturnpos_[7][1] := Ord('Z')-64;
 rotturnpos_[8][1] := Ord('Z')-64;

 rotturnpos_[1][2] := Ord('Q')-64;
 rotturnpos_[2][2] := Ord('E')-64;
 rotturnpos_[3][2] := Ord('V')-64;
 rotturnpos_[4][2] := Ord('J')-64;
 rotturnpos_[5][2] := Ord('Z')-64;
 rotturnpos_[6][2] := Ord('M')-64;
 rotturnpos_[7][2] := Ord('M')-64;
 rotturnpos_[8][2] := Ord('M')-64;

 AssignFile(F, 'rotors.txt');
 Reset(F);
 for j:=1 to 10 do
    begin
      ReadLn(F, s);
      readAlphabet(s, @rotor_[j]);
      invertRotor(j);
    end;
 CloseFile(F);

 WriteLn('Rotors:');
 for j:=1 to 10 do
   begin
     for i:=1 to 26 do
       Write(rotor_[j][i]);
     WriteLn;
   end;

 WriteLn('Inverted rotors:');
 for j:=1 to 10 do
   begin
     for i:=1 to 26 do
       Write(invrotor_[j][i]);
     WriteLn;
   end;


 AssignFile(F, 'reflectors.txt');
 Reset(F);
 for j:=1 to 4 do
    begin
      ReadLn(F, s);
      readAlphabet(s, @reflector_[j]);
    end;
 CloseFile(F);

 WriteLn('Reflectors:');
 for j:=1 to 4 do
   begin
     for i:=1 to 26 do
       Write(rotor_[j][i]);
     WriteLn;
   end;


 AssignFile(F, 'plugboard.txt');
 Reset(F);
 ReadLn(F, s);
 readAlphabet(s, @plugboard_);
 CloseFile(F);

 WriteLn('Plugboard:');
 for i:=1 to 26 do
   Write(plugboard_[i]);
 WriteLn;

 ciphertext_ := 'FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF';
 WriteLn('Ciphertext:');
 WriteLn(ciphertext_);
end;

function  TEnigmaApplication.isValidRotors(rot1, rot2, rot3, rot4 : Longint) : Boolean;
var valid : Boolean;
begin                // the roman numerals of the rotors are opposed to the odometric sense of Enigma!
 valid :=            (rot1>0) and (rot1<9);
 valid := valid and ((rot2>0) and (rot2<9));
 valid := valid and ((rot3>0) and (rot3<9));
 valid := valid and ((rot4=-1) or (rot4=9) or (rot4=10));      // only Beta or Gamma rotors
 valid := valid and (rot1<>rot2) and (rot1<>rot3) and (rot1<>rot4) and (rot2<>rot3) and (rot2<>rot4) and (rot3<>rot4);
 Result := valid;
end;

function  TEnigmaApplication.cipher(c : AnsiString; pos1, pos2, pos3, pos4, rot1, rot2, rot3, rot4, refl : Longint) : AnsiString;
var i : Longint;
    ch : Char;
    ok : Boolean;
    plaintext : AnsiString;
begin
  plaintext := '';
  i:=1;
  while (i<=length(c)) do
      begin
         nextPosition(pos1, pos2, pos3, rot1, rot2, rot3);
         ch := cipherChar(c, i, pos1, pos2, pos3, pos4, rot1, rot2, rot3, rot4, refl);
         if (ch=c[i]) then
             begin
                WriteLn('Internal error: encoded char never can be the entry char (a property of the machine)');
                WriteLn('Error for wheel positions '+IntToStr(pos3)+' '+IntToStr(pos2)+' '+IntToStr(pos1)+' ');
                WriteLn('Entry char '+c[i]);
                WriteLn('Exit char '+ch);
                ReadLn;
                Halt;

                Result := '';
                Exit;
             end;

         plaintext := plaintext+ch;
         Inc(i);
      end;

  Result := plaintext;
end;

procedure TEnigmaApplication.nextPosition(var pos1, pos2, pos3 : Longint; rot1, rot2, rot3 : Longint);
   function notchAligned(pa, rota : Longint) : Boolean;
   begin
     Result := ((pa-rotturnpos_[rota][1])=0) or ((pa-rotturnpos_[rota][2])=0);
     {Result := ((pa-rotturnpos_[rota][1] + pb-rotturnpos_[rotb][1]) = 0)  or
               ((pa-rotturnpos_[rota][2] + pb-rotturnpos_[rotb][2]) = 0); }
   end;
begin
 //odometric (?) change of first three rotor, 4th rotor is fixed
 Inc(pos1);
 if (pos1>25) then pos1:=0;
 if notchAligned(pos1, rot1) then
      begin
        Inc(pos2);
        if (pos2>25) then pos2:=0;
        if notchAligned(pos2, rot2) then
              begin
                Inc(pos3);
                if (pos3>25) then pos3:=0;
              end;
      end;
end;

function TEnigmaApplication.shiftPos(ch : Char; posrot: Longint) : Longint;
var res : Longint;
begin
 res := (ord(ch)-64+posrot);
 if (res<1) then res := res+26;
 if (res>26) then res := res - 26;
 Result := res;
end;

function  TEnigmaApplication.cipherChar(c : AnsiString; i, pos1, pos2, pos3, pos4, rot1, rot2, rot3, rot4, refl : Longint) : Char;
var plugch, r1ch, r2ch, r3ch, r4ch, reflch : Char;
begin
   plugch := plugboard_[Ord(c[i])-64];
   r1ch := rotor_[rot1][shiftPos(plugch, pos1)];
   r2ch := rotor_[rot2][shiftPos(r1ch, pos2-pos1)];
   r3ch := rotor_[rot3][shiftPos(r2ch, pos3-pos2)];

   if (rot4<>-1) then
        // naval version, 4th rotor in place
        begin
          r4ch := rotor_[rot4][shiftPos(r3ch, pos4-pos3)];
          reflch := reflector_[refl][shiftPos(r4ch, 26-pos4)];
        end
      else
        begin
          r4ch := r3ch;
          reflch := reflector_[refl][shiftPos(r4ch, 26-pos3)];
        end;
   if DEBUG then
     begin
       if isValidRotors(rot1, rot2, rot3, rot4) then
          WriteLn('Rotor combination is valid')
         else
          WriteLn('Invalid rotor combination!');
       WriteLn('Current position is : '+IntToStr(pos3)+' '+IntToStr(pos2)+' '+IntToStr(pos1));
       WriteLn('Entry char '+c[i]);
       WriteLn('After plugboard '+plugch);
       WriteLn('Entry char on first rotor is :'+Chr(64+shiftPos(plugch, pos1)));
       WriteLn('After 1st rotor '+r1ch);
       WriteLn('Entry char on second rotor is :'+Chr(64+shiftPos(r1ch, pos2-pos1)));
       WriteLn('After 2nd rotor '+r2ch);
       WriteLn('Entry char on third rotor is :'+Chr(64+shiftPos(r2ch, pos3-pos2)));
       WriteLn('After 3rd rotor '+r3ch);
       WriteLn('After 4th rotor '+r4ch);
       WriteLn('After reflector '+reflch);
     end;

   // going back now through the wheels
   if (rot4<>-1) then
      begin
          r4ch := invrotor_[rot4][shiftPos(reflch, pos4)];
          r3ch := invrotor_[rot3][shiftPos(r4ch, pos3-pos4)];
      end
      else
      begin
        r4ch := reflch;
        r3ch := invrotor_[rot3][shiftPos(r4ch, pos3)];
      end;

   r2ch := invrotor_[rot2][shiftPos(r3ch, pos2-pos3)];
   r1ch := invrotor_[rot1][shiftPos(r2ch, pos1-pos2)];
   plugch := plugboard_[shiftPos(r1ch, 26-pos1)];
   Result := plugch;

   if DEBUG then
     begin
       WriteLn('After 4th rotor '+r4ch);
       WriteLn('After 3rd rotor '+r3ch);
       WriteLn('After 2nd rotor '+r2ch);
       WriteLn('After 1st rotor '+r1ch);
       WriteLn('Exit char after plugboard '+plugch);
     end;
end;


procedure TEnigmaApplication.DoTest;
var pos1, pos2, pos3 : Longint;

  procedure encode(ch : Char);
  begin
    nextPosition(pos1,pos2,pos3,3,2,1);
    WriteLn(cipherchar(ch,1,pos1,pos2,pos3,-1,3,2,1,-1,1));
  end;

begin
  // remember to set DEBUG=true
  {
  // Test 1
  pos1:=0;
  pos2:=0;
  pos3:=0;
  WriteLn('Encoding ABCDEFGHIJKLMNOPQRSTUVXYZABCD with AAA, no wires on plugboard, rotors I, II, III');
  encode('A');
  encode('B');
  encode('C');
  encode('D');
  encode('E');
  encode('F');
  encode('G');
  encode('H');
  encode('I');
  encode('J');
  encode('K');
  encode('L');
  encode('M');
  encode('N');
  encode('O');
  encode('P');
  encode('Q');
  encode('R');
  encode('S');
  encode('T');
  encode('U');
  encode('V');
  encode('W');
  encode('X');
  encode('Y');
  encode('Z');
  encode('A');
  encode('B');
  encode('C');
  encode('D');
  encode('E');


  WriteLn('Result has to be BJELRQZVJWARXSNBXORSTNCFMEYYAQU');
  }

  //Test 2
  {
  WriteLn('Encoding HELLOWORLD with OBF, no wires on plugboard, rotors I, II, III');
  pos1:=Ord('F')-65;
  pos2:=Ord('B')-65;
  pos3:=Ord('O')-65;
  encode('H');
  encode('E');
  encode('L');
  encode('L');
  encode('O');
  encode('W');
  encode('O');
  encode('R');
  encode('L');
  encode('D');
  WriteLn('Result has to be QSTQPXUUDB');
  }

  // Test 3
  // Encode GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD with PVO, no wires on plugboard, rotors I, II, III');
  //WriteLn(cipher('GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD', Ord('O')-65, Ord('V')-65, Ord('P')-65, NONE, 3, 2, 1, NONE, 1));
  //WriteLn('Result has to be ');
  //WriteLn('FGIKWZAYFXONXETXZTGJSTDZNLQOICIZCASWIQEIF');

  // Encode GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD with PVO, wires A-T, E-N, rotors I, II, III');
  //WriteLn(cipher('GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD', Ord('O')-65, Ord('V')-65, Ord('P')-65, NONE, 3, 2, 1, NONE, 1));
  //WriteLn('Result has to be ');
  //WriteLn('FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF');

  // Encode FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF with PVO, wires A-T, E-N, rotors I, II, III');
  WriteLn(cipher('FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF', Ord('O')-65, Ord('V')-65, Ord('P')-65, NONE, 3, 2, 1, NONE, 1));
  WriteLn('Result has to be ');
  WriteLn('GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD');
  WriteLn('index of coincidence for GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD:');
  WriteLn(FloatToStr(calculateIndexOfCoincidence('GREETINGSFROMPOSCHIAVOTHECENTEROFTHEWORLD')));
  WriteLn('index of coincidence for FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF:');
  WriteLn(FloatToStr(calculateIndexOfCoincidence('FGRDDZMYFXOEXNAXZAGXSAWZNLKIGMIZCASYIQNIF')));
end;

function TEnigmaApplication.calculateIndexOfCoincidence(plain : AnsiString) : Extended;
var i, len, idx : Longint;
    sum : Extended;
begin
 // create frequency table first
 for i:=1 to 26 do
   freq_[i] := 0;
 len := length(plain);
 if len=0 then
     begin
       Result := 0;
       Exit;
     end;

 for i:=1 to len do
     begin
       idx := Ord(plain[i])-64;
       freq_[idx] := freq_[idx]+1;
     end;

 sum := 0;
 for i:=1 to 26 do
    begin
      if freq_[i]>0 then
        sum := sum + (freq_[i] * (freq_[i]-1))
    end;
 Result := 26 * sum/(len*(len-1));
end;

procedure TEnigmaApplication.outputValidPlainText(plain : AnsiString);
var F : TextFile;
    ioc : Extended;
begin
 ioc := calculateIndexOfCoincidence(plain);
 if ioc > bestioc_ then
     begin
        //bestioc_ := ioc; not so a good idea, better would be the one which comes closer to 1.73
        //WriteLn(plain);
        AssignFile(F, 'solutions.txt');
        Append(F);
        WriteLn(F, plain+' ioc:'+FloatToStr(ioc));
        CloseFile(F);
     end;
end;

procedure TEnigmaApplication.breakArmyEnigma(c : AnsiString);
var r1, r2, r3, r4, refl : Longint;
    p1, p2, p3 : Longint;
    plain : AnsiString;
begin
   WriteLn('Breaking Army or Commercial Enigma through brute force with known plugboard ...');
   bestioc_ := IOC_TRESHOLD;
   r4 := -1;
   for r1:=1 to 5 do
    for r2:=1 to 5 do
     for r3 :=1 to 5 do
      for refl:=1 to 2 do
        begin
          if not isValidRotors(r1, r2, r3, r4) then continue;
          WriteLn('Testing rotors/reflector '+IntToStr(r1)+' '+IntToStr(r2)+' '+IntToStr(r3)+' / '+IntToStr(refl));

           for p1:=0 to 25 do
             for p2:=0 to 25 do
               for p3:=0 to 25 do
                   begin
                      plain := cipher(c, p1, p2, p3, NONE, r1, r2, r3, r4, refl);
                      if length(plain)>0 then
                         OutputValidPlainText(plain);
                   end;
        end;
   WriteLn('Done, check solutions.txt :-)');
end;

procedure TEnigmaApplication.breakNavalEnigma(c : AnsiString);
var r1, r2, r3, r4, refl : Longint;
    p1, p2, p3, p4 : Longint;
    plain : AnsiString;
begin
   WriteLn('Breaking Naval Enigma through brute force with known plugboard ...');
   bestioc_ := IOC_TRESHOLD;
   for r4:=9 to 10 do
   for r1:=1 to 8 do
    for r2:=1 to 8 do
     for r3 :=1 to 8 do
      for refl:=3 to 4 do
        begin
          if not isValidRotors(r1, r2, r3, r4) then continue;
          WriteLn('Testing rotors/reflector '+IntToStr(r1)+' '+IntToStr(r2)+' '+IntToStr(r3)+' '+IntToStr(r4)+' / '+IntToStr(refl));

           for p1:=0 to 25 do
             for p2:=0 to 25 do
               for p3:=0 to 25 do
                for p4:=0 to 25 do
                   begin
                      plain := cipher(c, p1, p2, p3, p4, r1, r2, r3, r4, refl);
                      if length(plain)>0 then
                         OutputValidPlainText(plain);
                   end;
        end;
  WriteLn('Done, check solutions.txt :-)');
end;


procedure TEnigmaApplication.DoRun;
begin
  initEnigma;
  //doTest;
  //breakArmyEnigma(ciphertext_);
  breakNavalEnigma(ciphertext_);

  ReadLn;
  // stop program loop
  Terminate;
end;

constructor TEnigmaApplication.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
end;

destructor TEnigmaApplication.Destroy;
begin
  inherited Destroy;
end;


var
  Application: TEnigmaApplication;
begin
  Application:=TEnigmaApplication.Create(nil);
  Application.Title:='EnigmaApplication';
  Application.Run;
  Application.Free;
end.

