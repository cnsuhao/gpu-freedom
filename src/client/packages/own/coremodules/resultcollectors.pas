unit resultcollectors;
{
  This unit is charged to collect GPU res and make them available to other parts of the core
 
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses SyncObjs,
     gpuconstants, stacks;

type   
  TResultCollection = record
    jobId : String;
    
    startTime,
    totalTime: TDateTime;
    
    // circular buffers to store res
    resStr     : array[1..MAX_RESULTS_FOR_ID] of String;
    resFloat   : array[1..MAX_RESULTS_FOR_ID] of TStkFloat;
    isFloat    : array[1..MAX_RESULTS_FOR_ID] of Boolean;
    idx : Longint; // index of circular buffer
    
    N,                         // number of res
    N_float : Longint;         // number of single floats
    sum,                      // with sum and N_float we can compute average
    min,
    max,
    avg,
    variance,
    stddev    : TStkFloat;
	
    overrun : Boolean; // if there are more than MAX_RESULTS_FOR_ID
  end;
  
  
type TResultCollector = class(TObject)
  public
    constructor Create;
    destructor Destroy;
    
    function getResultCollection(jobId : String; var rescoll : TResultCollection) : Boolean;
    function registerResult(jobId : String; var stk : TStack) : Boolean;
  private
    // circular buffer to store result collections
    res     : array[1..MAX_RESULTS] of TResultCollection;    
    resIdx_ : Longint;
    CS_ : TCriticalSection;

    procedure initResultCollection(i : Longint);
    function findJobId(jobId : String) : Longint;

end;



implementation

constructor TResultCollector.Create;
var i : Longint;
begin
  inherited;
  for i:=1 to MAX_RESULTS do initResultCollection(i);
  resIdx_ := 0;
  CS_ := TCriticalSection.Create();
end;

destructor TResultCollector.Destroy;
begin
 CS_.Free;
 inherited;
end;

procedure TResultCollector.initResultCollection(i : Longint);
var j : Longint;
begin
  res[i].jobId := '';
  res[i].startTime := 0; //now;
  res[i].totalTime := 0;
  
  for j:=1 to MAX_RESULTS_FOR_ID do
    begin
      res[i].resStr[j] := '';
      res[i].resFloat[j] := 0;
      res[i].isFloat[j] := false;
    end;
  
  res[i].N := 0;
  res[i].N_float := 0;
  res[i].sum := 0;
  res[i].min := INF;
  res[i].max := -INF;
  res[i].avg := 0;
  res[i].stddev := 0; 
  res[i].overrun := false; 
end;

function TResultCollector.findJobId(jobId : String) : Longint;
var i : Longint;
begin
  Result := -1;
  for i:=1 to MAX_RESULTS do
    if res[i].jobId = jobId then
      begin
        Result := i;
        Exit;
      end;     
end;

function TResultCollector.getResultCollection(jobId : String; var rescoll : TResultCollection) : Boolean;
var i : Longint;
begin
  CS_.Enter;
  i := findJobId(jobId);
  CS_.Leave;
  if (i<0) then Result := false
  else
    begin
     Result := true;
     rescoll := res[i];
    end;
end;

function TResultCollector.registerResult(jobId : String; var stk : TStack) : Boolean;
var i, j : Longint;
begin
  Result := false;
  if stk.error.errorId>0 then Exit; // errors are not registered
  CS_.Enter;
  i := findJobId(jobId);
  if (i<0) then
         begin
           // this is a new job which needs registration
           Inc(resIdx_);
           if (resIdx_>MAX_RESULTS) then resIdx_:=1;
           i:=resIdx_;
           initResultCollection(i);
         end;
  
  // now we know that the job will be stored in res[i]  but where exactly? same game:
  Inc(res[i].idx);
  if res[i].idx>MAX_RESULTS_FOR_ID then res[i].idx:=1;
  
  // storing of the job
  res[i].resStr[res[i].idx] := stackToStr(stk);
  if (stk.Idx=1) and (stk.stkType[1]=FLOAT_STKTYPE) then
      begin
       res[i].resFloat[res[i].idx] := stk.stack[stk.Idx];
       res[i].isFloat[res[i].idx] := true;
      end
    else
      res[i].isFloat[res[i].idx] := false;
   
   // updating averages, sum, n, etc.
   Inc(res[i].N);
   res[i].overrun := (res[i].N>MAX_RESULTS_FOR_ID);
   
   res[i].N_float := 0;
   res[i].sum := 0;
   res[i].min := INF;
   res[i].max := -INF;
   for j:=1 to MAX_RESULTS_FOR_ID do
      if res[i].isFloat[j] then 
       begin 
        Inc(res[i].N_float);
        res[i].sum := res[i].sum + res[i].resFloat[j];
        if res[i].resFloat[j]<res[i].min then res[i].min:=res[i].resFloat[j];
        if res[i].resFloat[j]>res[i].max then res[i].max:=res[i].resFloat[j];        
       end;
   res[i].avg := res[i].sum/res[i].N_float;    
   
   
   res[i].variance := 0;
   // variance and standard deviation
   for j:=1 to MAX_RESULTS_FOR_ID do
      if res[i].isFloat[j] then 
         begin
           res[i].variance := res[i].variance + Sqr(res[i].resFloat[j]-res[i].avg);
         end;
   res[i].variance := res[i].variance/res[i].N_float;	 
   res[i].stddev := Sqrt(res[i].variance);
   CS_.Leave;
end;


end.
