program ioc;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp
  { you can add units after this };

const MAX_NB_SIZE = 65536;   // maximum representation for a char, from 0 to 65536
      MAX_KEY_LENGTH = 16;    // maximum number of index of coincidences to be computed, one for each key length
      MAX_CIPHERSIZE = 1000; // maximum ciphertext size
      DEBUG = false;
      NUMBER_OF_LETTERS = 26;


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
    ciphernb_,
    reducedcnb_  : Array[1..MAX_CIPHERSIZE] of Longint;
    cipherlength_,
    reducedlength_,
    max_nb_size_ : Longint;
    freqcount_   : Array[0..MAX_NB_SIZE] of Longint;

    procedure printError(errorMsg : String);
    procedure ReadCipherTextFromCharFile(filename : String);
    procedure ReadCipherTextFromNumberFile(filename : String);
    procedure processKeyLength(klength : Longint);
    function computeIocOnFrequencyTable : Extended;
    function computeChiTestOnFrequencyTable : Extended;
    procedure createFrequencyTableOnReducedCipherText;
    procedure printFrequencyTable;
    procedure printCipherText;
    procedure printReducedCipherText;
    procedure printStandardIOCs;
  end;


procedure TMyApplication.printError(errorMsg : String);
begin
  WriteLn('ERROR: '+errorMsg+' Exiting...');
  ReadLn;
  Halt;
end;

procedure TMyApplication.printStandardIOCs;
begin
WriteLn('IOCs for 26 chars');
WriteLn('English 	1.73');
WriteLn('French 	2.02');
WriteLn('German 	2.05');
WriteLn('Italian 	1.94');
WriteLn('Portuguese 	1.94');
WriteLn('Russian 	1.76');
WriteLn('Spanish 	1.94');
WriteLn('Random         1');

WriteLN('kappa-random: '+FloatToStr(1/NUMBER_OF_LETTERS));
WriteLn();
end;

procedure TMyApplication.printCipherText;
var i : Longint;
begin
 WriteLn('Ciphertext: '+ciphertext_);
 WriteLn('Numbered ciphertext: ');
 for i:=1 to cipherlength_ do
   Writeln(IntToStr(ciphernb_[i]));
end;

procedure TMyApplication.printReducedCipherText;
var i : Longint;
begin
 WriteLn('Reduced ciphertext: ');
 for i:=1 to reducedlength_ do
   Writeln(IntToStr(reducedcnb_[i]));
end;

procedure TMyApplication.printFrequencyTable;
var i : Longint;
begin
 WriteLn('Frequency table: ');
 WriteLn('-----------------------------');
 for i:=0 to max_nb_size_ do
   if freqcount_[i]>0 then
     WriteLn(IntToStr(i)+': '+IntToStr(freqcount_[i]));
  WriteLn('-----------------------------');
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

procedure TMyApplication.createFrequencyTableOnReducedCipherText;
var i, idx : Longint;
begin
  // reset frequency table counts
  for i:=0 to MAX_NB_SIZE do
     freqcount_[i] := 0;

  for i:=1 to reducedlength_ do
    begin
      idx := reducedcnb_[i];
      //WriteLn('Idx is '+IntTostr(idx));
      if (idx<0) or (idx>MAX_NB_SIZE) then printError('Internal error in createFrequencyTableOnReducedCipherText. Index was '+IntToStr(idx));
      freqcount_[idx] := freqcount_[idx] + 1;
    end;
end;

function TMyApplication.computeIocOnFrequencyTable : Extended;
var sum : Extended;
    i   : Longint;
begin
 sum := 0;
 for i:=0 to max_nb_size_ do
    begin
      if freqcount_[i]>0 then
        sum := sum + (freqcount_[i] * (freqcount_[i]-1))
    end;
 Result := sum/(reducedlength_*(reducedlength_-1));
end;

function TMyApplication.computeChiTestOnFrequencyTable : Extended;
var sum : Extended;
    i   : Longint;
begin
 sum := 0;
 for i:=0 to max_nb_size_ do
    begin
      if freqcount_[i]>0 then
        sum := sum + (freqcount_[i] * freqcount_[i]);
    end;
 Result := sum/(reducedlength_*reducedlength_);
end;

procedure TMyApplication.processKeyLength(klength : Longint);
var i, column, count : Longint;
    ioc       : Extended;
    star      : String;
begin
  if DEBUG then WriteLn('Processing key length '+IntToStr(klength)+'...');
  WriteLn;
  ioc := 0;
  count := 0;
  // we decompose the ciphertext in klenght columns and we compute
  // for each column the index of coincidence
  for column := 1 to klength do
     begin
       // init reducedcnb
       for i:=1 to MAX_CIPHERSIZE do reducedcnb_[i] := -1;
       reducedlength_ := cipherlength_ div klength;

       for i:=0 to reducedlength_-1 do
          begin
            reducedcnb_[i+1] := ciphernb_[i*klength+column];
          end;

       //if DEBUG then printReducedCipherText;
       createFrequencyTableOnReducedCipherText;
       if DEBUG then printFrequencyTable;
       if (klength=1) then
          WriteLn('      Chi Test: '+FloatToStr(computeChiTestOnFrequencyTable*NUMBER_OF_LETTERS));

       ioc := ioc + computeIocOnFrequencyTable;
       Inc(count);
     end;

  ioc := ioc / count;
  if (ioc*NUMBER_OF_LETTERS)>=1.7 then star := ' ***' else star := '    ';

  WriteLn('Index of Coincidence for key length '+IntToStr(klength));
  //WriteLn('       kappa-plaintext: '+FloatToStr(ioc));
  WriteLn('      IOC for 26 chars: '+FloatToStr(ioc*NUMBER_OF_LETTERS)+star);
  WriteLn();
  if DEBUG then WriteLn('Processing key length '+IntToStr(klength)+' over.');
end;

procedure TMyApplication.DoRun;
var
  ErrorMsg: String;
  i       : Longint;
begin
  filename_   := ParamStr(1);
  if Trim(filename_) = '' then printError('No file name specified. ');
  if not FileExists(filename_) then printError('Could not find '+filename_+'.');
  printStandardIOCs;

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

  for i:=1 to MAX_KEY_LENGTH do processKeyLength(i);
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

