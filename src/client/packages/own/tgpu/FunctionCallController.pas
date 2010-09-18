unit FunctionCallController;
//TODO: adding a critical section blocks the
//      core, why?
{ $DEFINE CRITICALSECTION}
{
 The object defined in this unit makes sure
 that only one function with a given name is called
 by one thread at a time. Two threads cannot be
 in the same DLL function. This should increase stability
 of the plugin mechanism
}
interface
uses SysUtils, Classes, SyncObjs;

type TFunctionCallController = class(TObject)
    constructor Create(NbOfThreads : Integer);
    destructor Destroy;override;
    function isAlreadyCalled(Name : String) : Boolean;
    procedure registerFunctionCall(Name : String; pluginName  : String; ThreadId : Longint);
    procedure unregisterFunctionCall(ThreadId : Longint);
    // following function takes a GPU command and verifies that
    // it does not have an active DLL call as a substring
    function verifyCommandForConflicts(Command : String) : Boolean;

    // this function is used to remember functions, which are
    // allowed to run concurrently despite the mechanism implemented
    // in this unit.
    // terragen function is such an example
    procedure allowRunningFunctionConcurrently(Name : String);
    // clears contents, for example needed if user wants to reset
    // the virtual machine
    procedure clear();

    function getFunctionName(i : Integer) : String;
    function getPluginName(i : Integer) : String;
   private
    MaxThreads : Longint;
    ConcurrentlyAllowedFunctions : TStringList;
    ThreadFunctionNames : Array of String;
    PluginNames : Array of String;
    {$IFDEF CRITICALSECTION}  
    CS     : TCriticalSection;
    {$ENDIF}

end;

implementation


constructor TFunctionCallController.Create(NbOfThreads : Integer);
begin
 inherited Create;
 MaxThreads := NbOfThreads;
 SetLength(ThreadFunctionNames, NbOfThreads);
 SetLength(PluginNames, NbOfThreads);
 ConcurrentlyAllowedFunctions := TStringList.Create;
 {$IFDEF CRITICALSECTION}
 if not Assigned(CS) then CS := TCriticalSection.Create;
 {$ENDIF}
end;


destructor TFunctionCallController.Destroy;
begin
 ConcurrentlyAllowedFunctions.Free;
 {$IFDEF CRITICALSECTION}
 if Assigned(CS) then CS.Free;
 {$ENDIF}
 inherited Destroy;
end;

function TFunctionCallController.isAlreadyCalled(Name : String) : Boolean;
var i : Longint;
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 isAlreadyCalled := False;
 for i:=0 to MaxThreads-1 do
  if ThreadFunctionNames[i] = Name then
        begin
          isAlreadyCalled := True;
          Exit;
        end;
 {$IFDEF CRITICALSECTION}
 CS.Leave;
 {$ENDIF}
end;

procedure TFunctionCallController.registerFunctionCall(Name : String; PluginName : String; ThreadId : Longint);
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 // exception to the rule: if the function is allowed to run
 // concurrently, we simply do not register the function, such that the
 // FunctionCallController is out of order for that particular functions
 if ConcurrentlyAllowedFunctions.IndexOf(Name)>-1 then Exit;

 // we register the function in the array here
 ThreadFunctionNames[ThreadId-1] := Name;
 PluginNames[ThreadId-1] := PluginName;

 {$IFDEF CRITICALSECTION}
 CS.Leave;
 {$ENDIF}
end;

procedure TFunctionCallController.unregisterFunctionCall(ThreadId : Longint);
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 ThreadFunctionNames[ThreadId-1] := '';
 PluginNames[ThreadId-1] := '';
 {$IFDEF CRITICALSECTION}
 CS.Leave;
 {$ENDIF}
end;

function TFunctionCallController.verifyCommandForConflicts(Command : String) : Boolean;
var i : Longint;
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 Result := False; // in the beginning we assume no conflicts
 for i:=0 to MaxThreads-1 do
   begin
    if Trim(ThreadFunctionNames[i]) <> '' then
         begin
          if Pos(ThreadFunctionNames[i], Command) > 0 then
             begin
              // a thread is currently inside a function of a DLL
              // this command would like to call
              Result := True;
              Exit;
             end;
         end;
   end;
 {$IFDEF CRITICALSECTION}
  CS.Leave;
  {$ENDIF}
end;

procedure TFunctionCallController.allowRunningFunctionConcurrently(Name : String);
begin
 ConcurrentlyAllowedFunctions.Add(Name);
end;

procedure TFunctionCallController.Clear;
var i : Longint;
begin
 {$IFDEF CRITICALSECTION}
  CS.Enter;
 {$ENDIF}
  for i:=0 to MaxThreads-1 do
   begin
    ThreadFunctionNames[i] := '';
    PluginNames[i] := '';
   end;
 {$IFDEF CRITICALSECTION}
  CS.Leave;
  {$ENDIF}
end;

function TFunctionCallController.getFunctionName(i : Integer) : String;
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 if ((i>0) and (i<=MaxThreads)) then
    Result := ThreadFunctionNames[i-1]
   else
    Result := '';
 {$IFDEF CRITICALSECTION}
 CS.Leave;
 {$ENDIF}
end;

function TFunctionCallController.getPluginName(i : Integer) : String;
begin
 {$IFDEF CRITICALSECTION}
 CS.Enter;
 {$ENDIF}
 if ((i>0) and (i<=MaxThreads)) then
    Result := PluginNames[i-1]
   else
    Result := '';
 {$IFDEF CRITICALSECTION}
 CS.Leave;
 {$ENDIF}
end;

end.
