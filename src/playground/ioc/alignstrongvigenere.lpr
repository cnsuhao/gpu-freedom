program alignstrongvigenere;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp
  { you can add units after this };

const MAX_NB_SIZE = 10000;   // maximum representation for a char, from 0 to 65536
      MAX_CIPHERSIZE = 1000; // maximum ciphertext size
      MAX_KEYLENGTH = 15;
      DEBUG = true;


type

  { TMyApplication }

  TMyApplication = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
    procedure WriteHelp; virtual;
  private
    filename_    : String;
    ciphertext_  : AnsiString;

    ciphernb_    : Array[1..MAX_CIPHERSIZE] of Longint;
    reducedcnb_  : Array[1..MAX_KEYLENGTH] of Array[1..MAX_CIPHERSIZE] of Longint;
    cipherlength_,
    reducedlength_,
    klength_,
    max_nb_size_ : Longint;
    freqcount_   : Array[1..MAX_KEYLENGTH]of Array [0..MAX_NB_SIZE] of Longint;
    shift_       : Array[1..MAX_KEYLENGTH] of Longint;

    procedure printError(errorMsg : String);
    procedure printCipherText;
    procedure ReadCipherTextFromNumberFile(filename : String);
    procedure ReadCipherTextFromCharFile(filename : String);
    procedure processKeyLength(klength : Longint);
    procedure createFrequencyTableOnReducedCipherText(column : Longint);
    function  computeChiTestOnFrequencyTable(column1, column2 : Longint) : Extended;
    procedure shiftReducedCipherTextByOne(column : Longint);
    procedure adaptCipherTextToShift;
  end;


procedure TMyApplication.printError(errorMsg : String);
begin
  WriteLn('ERROR: '+errorMsg+' Exiting...');
  ReadLn;
  Halt;
end;

procedure TMyApplication.printCipherText;
var i : Longint;
begin
 WriteLn('Ciphertext: '+ciphertext_);
 WriteLn('Numbered ciphertext: ');
 for i:=1 to cipherlength_ do
   Writeln(IntToStr(ciphernb_[i]));
end;

procedure TMyApplication.ReadCipherTextFromCharFile(filename : String);
var F   : TextFile;
    str : AnsiString;

    i : Longint;
begin
  ciphertext_ := '';
  AssignFile(F, filename);
  Reset(F);
  While not EOF(F) do
     begin
       ReadLn(F, str);
       ciphertext_ := ciphertext_ + str;
     end;
  CloseFile(F);

  // now converting ciphertext into ciphernb_ representation
  cipherlength_ := length(ciphertext_);
  for i:=1 to cipherlength_ do
     begin
       ciphernb_[i] := Ord(ciphertext_[i]);
     end;

end;

procedure TMyApplication.ReadCipherTextFromNumberFile(filename : String);
var F : TextFile;
    Str : String;
    i   : Longint;
begin
  AssignFile(F, filename);
  Reset(F);
  i := 0;
  While not EOF(F) do
     begin
       ReadLn(F, str);
       Inc(i);
       ciphernb_[i] := StrToIntDef(str, -1);
       if (ciphernb_[i]) > MAX_NB_SIZE then
                begin
                   CloseFile(F);
                   printError('Number is too big '+IntToStr(ciphernb_[i])+'>'+IntToStr(MAX_NB_SIZE));
                end;
       if (ciphernb_[i]) < 0 then
                begin
                   CloseFile(F);
                   printError('Could not parse number or it is a negative number! ('+str+')');
                end;
     end;
  CloseFile(F);
end;


procedure TMyApplication.createFrequencyTableOnReducedCipherText(column : Longint);
var i, idx : Longint;
begin
  // reset frequency table counts
  for i:=0 to MAX_NB_SIZE do
     begin
      freqcount_[column][i] := 0;
     end;

  for i:=1 to reducedlength_ do
    begin
      idx := reducedcnb_[column][i];
      //WriteLn('Idx is '+IntTostr(idx));
      if (idx<0) or (idx>MAX_NB_SIZE) then printError('Internal error in createFrequencyTableOnReducedCipherText. Index was '+IntToStr(idx));
      freqcount_[column][idx] := freqcount_[column][idx] + 1;
    end;
end;


function TMyApplication.computeChiTestOnFrequencyTable(column1, column2 : Longint) : Extended;
var sum : Extended;
    i   : Longint;
begin
 sum := 0;
 for i:=0 to max_nb_size_ do
    begin
      if (freqcount_[column1][i]>0) and (freqcount_[column2][i]>0) then
        sum := sum + (freqcount_[column1][i] * freqcount_[column2][i]);
    end;
 Result := sum/(reducedlength_*reducedlength_);
end;



procedure TMyApplication.processKeyLength(klength : Longint);
var i, shift, column, count, bestshift : Longint;
    chi, bestchi : Extended;
begin
  if DEBUG then WriteLn('Processing key length '+IntToStr(klength)+'...');
  WriteLn;

  // we decompose the ciphertext in klenght columns
  reducedlength_ := cipherlength_ div klength;
  for column := 1 to klength do
     begin
       // init reducedcnb
       for i:=1 to MAX_CIPHERSIZE do reducedcnb_[column][i] := -1;

       for i:=0 to reducedlength_-1 do
          begin
            reducedcnb_[column][i+1] := ciphernb_[i*klength+column];
          end;
    end;



  createFrequencyTableOnReducedCipherText(1);
  shift_[1] := 0;

  for column:=2 to klength do
   begin
     bestchi := 0;
     bestshift := 0;
     for shift:=0 to 26 do  // to 26 so that the ciphertext stays aligned for later use
      begin
        if (shift>0) then shiftReducedCipherTextByOne(column);
        createFrequencyTableOnReducedCipherText(column);
        chi := computeChiTestOnFrequencyTable(1,column);
        //WriteLn('chi-test: '+FloatToStr(chi)+' for shift '+IntToStr(shift));
        if chi>bestchi then
             begin
               bestchi := chi;
               bestshift := shift;
             end;
      end;
      shift_[column] := bestshift;
      WriteLn('For column '+IntToStr(column)+' the best shift was '+IntToStr(bestshift)+' with value '+FloatToStr(bestchi));
    end;

  if DEBUG then WriteLn('Processing key length '+IntToStr(klength)+' over.');
end;


procedure TMyApplication.shiftReducedCipherTextByOne(column : Longint);
var i : Longint;
    cipherchar : Longint;
begin
 for i:=1 to reducedlength_ do
    begin
      cipherchar := reducedcnb_[column][i] + 1;
      if cipherchar>90 then cipherchar := 65;
      reducedcnb_[column][i] := cipherchar;
    end;
end;


procedure TMyApplication.adaptCipherTextToShift;
var column,shift,i : Longint;
    ciphertext : AnsiString;
begin
 for column:=1 to klength_ do
   for shift:=1 to shift_[column] do
      begin
        shiftReducedCipherTextByOne(column);
      end;

 //now it is time to print the adapted ciphertext :-D
 ciphertext := '';
 for i:=1 to reducedlength_ do
   for column:=1 to klength_ do
    begin
      ciphertext := ciphertext + Chr(reducedcnb_[column][i]);
    end;
 WriteLn;
 WriteLn('The adapted ciphertext wich could be a simple Caesar, or a more complex substitution is: ');
 WriteLn(ciphertext);
end;

procedure TMyApplication.DoRun;
var
  ErrorMsg: String;
begin
  filename_   := ParamStr(1);
  if Trim(filename_) = '' then printError('No file name specified as first parameter. ');
  if not FileExists(filename_) then printError('Could not find '+filename_+'.');
  klength_ := StrToIntDef(ParamStr(2),0);
  if klength_ = 0 then printError('Key length needs to be defined as second parameter');

  // parse parameters
  if HasOption('n','number') then
        begin
          max_nb_size_ := MAX_NB_SIZE;
          WriteLn('Reading from file with list of numbers separated by end of line... ('+filename_+', max number size '+IntToStr(max_nb_size_)+')');
          ReadCipherTextFromNumberFile(filename_);
        end
       else
        begin
          max_nb_size_ := 255;
          WriteLn('Reading standard ciphertext file, in text or binary format... ('+filename_+', max char size '+IntToStr(max_nb_size_)+')');
          ReadCipherTextFromCharFile(filename_);
        end;

  if DEBUG then printCipherText;

  processKeyLength(klength_);
  adaptCipherTextToShift;
  ReadLn;
  // stop program loop
  Terminate;
end;

constructor TMyApplication.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
end;

destructor TMyApplication.Destroy;
begin
  inherited Destroy;
end;

procedure TMyApplication.WriteHelp;
begin
  { add your help code here }
  writeln('Usage: ',ExeName,' -h');
end;

var
  Application: TMyApplication;
begin
  Application:=TMyApplication.Create(nil);
  Application.Title:='My Application';
  Application.Run;
  Application.Free;
end.

