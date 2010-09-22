unit resultcollectors;
{
  This unit is charged to collect GPU results and make them available to other parts of the core
 
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses gpuconstants, stacks;

type   
  TResultCollection = record
    jobId : String;
    
    startTime,
    totalTime: TDateTime;
    
    // circular buffers to store results
    resultsStr    : array[1..MAX_RESULTS_FOR_ID] of String;
    resultFloat   : array[1..MAX_RESULTS_FOR_ID] of TGPUFloat;
    isSingleFloat : array[1..MAX_RESULTS_FOR_ID] of Boolean;
    idx : Longint; // index of circular buffer
    
    N                         // number of results
    N_float : Longint         // number of single floats
    sum,                      // with sum and N_float we can compute average
    min,
    max,
    avg,
    stddev    : TGPUFloat;  
  end;
  
  
type TResultCollector = class(TObject)
  public
    constructor Create;
    destructor Destroy;
    
    function getResultCollection(jobId : String; var res : TResultCollection) : Boolean;
    function registerResult(jobId : String; var stk : TStack; var error : TGPUError) : Boolean;
  private
    // circular buffer to store result collections
    results : array[1..MAX_RESULTS] of TResultCollection;    
    resultsIdx_ : Longint;
    
    function initResultCollection(i : Longint);
    function findJobId(jobId : String) : Longint;
    CS_ : TCriticalSection;
end;



implementation

constructor TResultCollector.Create;
var i : Longint;
begin
  inherited;
  for i:=1 to MAX_RESULTS do initResultCollection(i);
  resultsIdx := 0;
  CS_ := TCriticalSection.Create();
end;

destructor TResultCollector.Destroy;
begin
 CS_.Free;
 inherited;
end;

function TResultCollector.initResultCollection(i : Longint);
var j : Longint;
begin
  results[i].jobId := '';
  results[i].startTime := now;
  results[i].totalTime := 0;
  
  for j:=1 to MAX_RESULTS_FOR_JOB_ID do
    begin
      resultsStr[j] := '';
      resultsFloat[j] := 0;
      isSingleFloat[j] := false;
    end;
  
  results[i].N := 0;
  results[i].sum := 0;
  results[i].min := INF;
  results[i].max := -INF;
  results[i].avg := 0;
  results[i].stddev := 0; 
end;

function TResultCollector.findJobId(jobId : String) : Longint;
var i : Longint;
begin
  Result := -1;
  for i:=1 to MAX_RESULTS do
    if results[i].jobId := jobId then
      begin
        Result := i;
        Exit;
      end;     
end;

function getResultCollection(jobId : String; var res : TResultCollection) : Boolean;
var i : Longint;
begin
  CS_.Enter;
  i := findJobId(jobId);
  CS_.Leave;
  if (i<0) then Result := false
  else
    begin
     Result := true;
     res := results[i];
    end;
end;

function registerResult(jobId : String; var stk : TStack; var error : TGPUError) : Boolean;
var i, j : Longint;
begin
  Result := false;
  if error.errorId>0 then Exit; // errors are not registered
  CS_.Enter;
  i := findJobId(jobId);
  if (i<0) then
         begin
           // this is a new job which needs registration
           Inc(resultsIdx_);
           if (resultsIdx_>MAX_RESULTS) then resultsIdx_:=1;
           i:=resultsIdx;
           initResultCollection(i);
         end;
  
  // now we know that the job will be stored in results[]i  but where exactly? same game:
  Inc(results[i].idx);
  if results[i].idx>MAX_RESULTS_FOR_JOB_ID then results[i].idx:=1;
  
  // storing of the job
  results[i].resultStr := stackToStr(stk);
  if (stk.Idx=1) and (stk.stkType(1)=GPU_FLOAT_STKTYPE) then
      begin
       results[i].resultFloat := stk.stack[stk.Idx];
       results[i].isSingleFloat := true;
      end;
    else
      results[i].isSingleFloat := false;    
   
   // updating averages, sum, n, etc.
   Inc(results[i].N);
   if (results[i].N>MAX_RESULTS_FOR_JOB_ID) then results[i].N:=MAX_RESULTS_FOR_JOB_ID;
   
   results[i].N_float := 0;
   results[i].sum := 0;
   results[i].min := INF;
   results[i].max := -INF;
   for j:=1 to MAX_RESULTS_FOR_JOB_ID do
      if results[i].isSingleFloat[j] then 
       begin 
        Inc(results[i].N_float);
        results[i].sum := results[i].sum + results[i].resultFloat[j];
        if results[i].resultFloat[j]<results[i].min then results[i].min:=results[i].resultFloat[j];
        if results[i].resultFloat[j]>results[i].max then results[i].max:=results[i].resultFloat[j];        
       end;
   results[i].avg := results[i].sum/results[i].N_float;    
   
   //TODO: implement calculation of standard deviation   
   CS_.Leave;
end;


end.