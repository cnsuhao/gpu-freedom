unit ComputationThread;
{
    ComputationThread is a thread which executes stored in
    JobForThread.StackCommands.
    
    In the method Execute following things happen:
    
    1. The procedure StackToResult parses JobForThread.StackCommands
    in the following way:
    The routine starts with an empty TStack, defined in definitions unit
    and called here JobForThread.Stack.
    Numbers or string (strings are enclosed by quotes) are loaded on the TStack object.
    Strings which are not enclosed by quotes are interpreted as DLL function calls.
    To call the DLL functions, the PlugMan object is used. DLL functions can create
    or crunch parameters on the TStack object, while performing some useful
    kind of computation.
    
    Special commands are handled inside a subroutine of StackToResult.
    
    2.  The stack is converted back to a string by using the StackToStr function
         The resulting conversion is in JobForThread.Result. GPU takes the result
         and distributes it to frontends.
    

}

interface

uses
  Classes, Definitions, SysUtils, PluginManager, Jobs, Math,
  gpu_utils, FunctionCallController, common;

const
  WAIT_FOR_RESULT = 3000;
{how much threads sleep waiting for a result using first[], avg[] and so on}

type
  TNotifyEvent = procedure(Sender: TObject) of object;


{the threads which will take computational jobs and interpret them}
type
  TComputationThread = class(TThread)

    PlugMan:      TPluginManager;
    {the plugin manager is used to search for functions into plugins}
    JobForThread: TJobOnGnutella;
    {these variables are used to pass the job and the stack to the thread}

    JobDone: boolean;
    {if the thread is finished, then it sets JobDone to true}

    Plugin:    THandle;
    UpdateNeeded: boolean;  {externally set by the GPU main form}
    FormatSet: TFormatSet;

    GPUID: TGPUIdentity; {holds identity information}

    MyThreadID: TThreadID;

    // this is unique object for all threads and helps avoiding
    // two threads into the same DLL function for stability reasons.
    FuncController : TFunctionCallController;
    // this is the ID for the FunctionCallController
    FuncThreadID : Longint;

  private
    {the array used to pass PChar pointers to dlls}
    StrArray: array[1..MAXSTACK] of string;
    function StackToStr(var Stk: TStack): string;

  protected
    //constructor Create(CreateSuspended: Boolean);
    procedure Execute; override;
    procedure StackToResult(var J: TJob; var Stk: TStack);

    procedure Update;
    procedure SendResultCallback(stk: PStack);

    procedure CleanStackMem(var stk: TStack);

    //some synchronize methods
    procedure SyncOnJobCreated;
    procedure SyncOnJobFinished;

  end;


procedure SendCallBack(stk: PStack); stdcall;
procedure SendStack; stdcall;

var
  tl: TThreadList;
  //following variables are used to know the
  //traffic in download and upload on the current node
  //to adapt plugin traffic on the network
  ct_traffic_download, ct_traffic_upload: double;

implementation

function LoadStringOnStack(str : String; var Stk : TStack) : Boolean;
var Idx : Longint;
begin
  Result := False;
  Idx := Stk.StIdx;
  Inc(Idx);
  if Idx > MAXSTACK then Exit;
  Stk.StIdx := Idx;
  Stk.Stack[Stk.StIdx]      := INF;
  Stk.PCharStack[Stk.StIdx] := StringToPChar(Str);
  Result := True;
end;

function LoadExtendedOnStack(ext : Extended; var Stk : TStack) : Boolean;
var Idx : Longint;
begin
  Result := False;
  Idx := Stk.StIdx;
  Inc(Idx);
  if Idx > MAXSTACK then Exit;
  Stk.StIdx := Idx;
  Stk.Stack[Stk.StIdx]      := ext;
  if Assigned(Stk.PCharStack[Stk.StIdx]) then FreeAndNil(Stk.PCharStack[Stk.StIdx]);
  Result := True;
end;

function LoadBooleanOnStack(b : boolean; var Stk : TStack) : Boolean;
var Idx : Longint;
begin
  Result := False;
  Idx := Stk.StIdx;
  Inc(Idx);
  if Idx > MAXSTACK then Exit;
  Stk.StIdx := Idx;
  if (b) then
     Stk.Stack[Stk.StIdx]      := 1
  else
     Stk.Stack[Stk.StIdx]      := 0;
  if Assigned(Stk.PCharStack[Stk.StIdx]) then FreeAndNil(Stk.PCharStack[Stk.StIdx]);
  Result := True;
end;



 {********************************************************************}
 {*                                                                  *}
 {* TComputationThread methods                                       *}
 {*                                                                  *}
 {********************************************************************}
 //  **** french comments
 //  TComputationThread est la d‚finition des threads qui calcule des choses
 //  ‚crites dans le champ edit "stackcommands" (commandes de pile) dans le
 //  Record TJob.
 //  Dans cet objet la fonction StackToResult prend la chaŒne et l'interprŠte:
 //  Si le nombre est trouv‚ il est simplement charg‚ dans la pile.
 //  Si une commande telle que "add" est trouv‚e, il y aura une recherche dans
 //  tous les plugin d'une commande nomm‚e add. La procedure add sera appel‚e
 //  et la structure entiŠre de la pile sera pass‚e en paramŠtre.

 //  Noter que cette fonction peut aussi ˆtre r‚cursive, dans le cas ou des
 //  accolades comme celles de l'exemple ci-dessous sont trouv‚es

 //  par exemple il y a une commande du type:    0,{1,add},10,rpt
 //  qui va ajouter … 0 pour dix foix le nombre 1
 //  ****** english original one
 //  TComputationThread is the definition of threads which compute things written
 //  in the edit field "StackCommands", in Record TJob.
 //  In this object the function StackToResult takes the string, and interpretes
 //  it: if a number is found it is simply loaded in the TStack structure. If a
 //  command like "add" is found, there will be a search in all plugins for a
 //  function called add. Add will be called and the whole TStack structure is
 //  passed as parameter.

 //  Notice that the function can be also recursive, in the case that brackets
 //  like this are found .

 //  For example there is a command 0,{1,add},10,rpt which will add to 0 for
 //  ten times the number one.


 //constructor TComputationThread.Create(CreateSuspended: Boolean);
 //begin
 //        inherited Create(CreateSuspended);

 //        //JobForThread.Stack.SendCallback := @self.SendResultCallback;
 //end;


{this function is the main thread function}
procedure TComputationThread.Execute;
begin
  with TL.LockList do
    Add(Self);
  TL.UnlockList;
  try
    MyThreadID := GetCurrentThreadID;

    JobDone   := False;
    FormatSet := TFormatSet.Create;
    try
      SyncOnJobCreated; //removed synchronize call as it had problems
      try
        JobForThread.Stack.Thread     := Self;
        JobForThread.Stack.SendCallback := @SendCallBack;
        JobForThread.Stack.SendStack  := @SendStack;
        JobForThread.Stack.GpuGetMem  := @GpuGetMem;
        JobForThread.Stack.GpuReallocMem := @GpuReallocMem;
        JobForThread.Stack.GpuFreeMem := @GpuFreeMem;
        {compute the result using the stack mechanism}
        StackToResult(TJob(JobForThread), JobForThread.Stack);
        JobForThread.Result := StackToStr(JobForThread.Stack);
        JobForThread.ComputedTime := Time - JobForThread.ComputedTime;
        //JobDone := True;
      except
        on E: Exception do
          JobForThread.Result :=
            '''Exception caught: ' + StringReplace(E.Message, '''',
            '"', [rfReplaceAll]) + '''';
      end;
      SyncOnJobFinished;
      //clean up PChar's that may be left in stack
      CleanStackMem(JobForThread.Stack);
      JobDone := True;
      //if FormatSet <> nil then FormatSet.Free;
    finally
      FormatSet.Free;
    end;
  finally
    with TL.LockList do
    begin
      if IndexOf(Self) >= 0 then //should be
        Delete(IndexOf(Self));
    end;
    TL.UnlockList;
  end;
end;

procedure TComputationThread.SyncOnJobCreated;
begin
  try
    if not Terminated and Assigned(JobForThread.OnCreated) then
      JobForThread.OnCreated(JobForThread);
  except
  end;
end;

procedure TComputationThread.SyncOnJobFinished;
begin
  try //maybe a overkill, but make sure thread continues.
    if not Terminated and Assigned(JobForThread.OnFinished) then
      JobForThread.OnFinished(JobForThread);
  except
  end;
end;

procedure TComputationThread.StackToResult(var J: TJob; var Stk: TStack);
var
  Arg:    string;        {the current argument inside the stack: a number, a string or a command}
  pluginName : String;   {the name of the currently used plugin}
  tmp        : String;   {used as temporary variable}
  JobToRepeat: TJob;
  countL: longword;      {used for montecarlo packets}
  ArgFound,              {if argument is not a number, ArgFound is true}
  ResultFunc: boolean;   {ResultFunc is the result of the Dll function}

  theFunction: PDLLFunction;
  i: integer;

  function HandleSpecialCommands(var S: string): boolean;
    {this functions handles special commands that cannot be moved to plugins}
  var tmpSpecial : String;
  
        
  begin
    Result := False;

    if Pos('{', S)=1 then
    begin
      Result := True;
      Delete(S, 1, 1);
      JobToRepeat.StackCommands := Copy(S, 1, Length(S) - 1);
    end
    else
    if Pos('rpt', S)=1 then
    begin
      Result := True;
      if Stk.StIdx < 1 then
        Exit;
      countL    := Trunc(Stk.Stack[Stk.StIdx]);
      Stk.StIdx := Stk.StIdx - 1;

      S := JobToRepeat.StackCommands;

      repeat
                                {now recursive call to StackResult with same stack
                                 but different job}
        JobToRepeat.StackCommands := S;

        StackToResult(JobToRepeat, Stk);
        countL := countL - 1;
      until countL <= 0;
      {result after repeat is in the stack at StIdx position}
    end
    else
    if 'sleep' = Arg then
    begin
      Result := True;
      if Stk.StIdx > 0 then
      begin
        Sleep(Round(Stk.Stack[Stk.StIdx] * 1000));
        Stk.StIdx := Stk.StIdx - 1;
      end;
    end
    else
    if 'nodename'= Arg then Result := LoadStringOnStack(MyGPUID.NodeName, Stk)
    else
    if 'team' = Arg then Result := LoadStringOnStack(MyGPUID.Team, Stk)
    else
    if 'country' = Arg then Result := LoadStringOnStack(MyGPUID.Country, Stk)
    else
    if 'nodeid' = Arg then Result := LoadStringOnStack(MyGPUID.NodeId, Stk)
    else
    if 'uptime' = Arg then Result := LoadExtendedOnStack(MyGPUID.Uptime, Stk)
    else
    if 'totuptime' = Arg then Result := LoadExtendedOnStack(MyGPUID.TotalUptime, Stk)
    else
    if 'opsys' = Arg then Result := LoadStringOnStack(MyGPUID.OS, Stk)
    else
    if ('ip' = Arg) or ('remoteip' = Arg)  then
       Result := LoadStringOnStack(MyGPUID.IP, Stk)
    else
    if ('port' = Arg) then Result := LoadExtendedOnStack(MyGPUID.Port, Stk)
    else
    if ('cputype' = Arg) then Result := LoadStringOnStack(MyGPUID.Processor, Stk)
    else
    if ('version' = Arg) then Result := LoadStringOnStack(MyGPUID.Version, Stk)
    else
    if ('mhz' = Arg) then Result := LoadExtendedOnStack(MyGPUID.SpeedMHz, Stk)
    else
    if ('ram' = Arg) then Result := LoadExtendedOnStack(MyGPUID.RAM, Stk)
	else
    {
    if ('memused' = Arg) then Result := LoadExtendedOnStack(memused, Stk)
    else
     if ('memtotalspace' = Arg) then Result := LoadExtendedOnStack(memtotalspace, Stk)
    else
     if ('memoverhead' = Arg) then Result := LoadExtendedOnStack(memoverhead, Stk)
    else
     if ('memheaperrorcode' = Arg) then Result := LoadExtendedOnStack(memheaperrorcode, Stk)
    else }
    if ('acceptincoming' = Arg) then Result := LoadBooleanOnStack(MyGPUId.AcceptIncoming, Stk)
    else
    if ('loadedplugins' = Arg) then
            begin
               Result := false;
               PlugMan.PluginNamesTop;
               repeat
                tmp := PlugMan.GetPluginsName;
                if tmp <> '' then LoadStringOnStack(tmp, Stk);
               until tmp = '';
               Result := true;
            end
    else
    if ('loaddll' = Arg) then
    begin
      Result := PlugMan.LoadSinglePlugin(StrPas(Stk.PCharStack[Stk.StIdx]));
      Stk.PCharStack[Stk.StIdx] := nil;
      if Result then
        stk.Stack[stk.StIdx] := 1
      else
        stk.Stack[stk.StIdx] := 0;
    end
    else
    if ('unloaddll' = Arg) then
    begin
      Result := PlugMan.UnloadSinglePlugin(StrPas(Stk.PCharStack[Stk.StIdx]));
      Stk.PCharStack[Stk.StIdx] := nil;
      if Result then
        stk.Stack[stk.StIdx] := 1
      else
        stk.Stack[stk.StIdx] := 0;
    end
    else
    if ('xyz_netmapper' = Arg) then
    begin
      Result := True;
      Stk.StIdx := Stk.StIdx+3;
      Stk.Stack[Stk.StIdx]      := MyGPUID.netmap_x;
      Stk.Stack[Stk.StIdx-1]    := MyGPUID.netmap_y;
      Stk.Stack[Stk.StIdx-2]    := MyGPUID.netmap_z;
      Stk.PCharStack[Stk.StIdx] := nil;
      Stk.PCharStack[Stk.StIdx-1] := nil;
      Stk.PCharStack[Stk.StIdx-2] := nil;
    end
    else
    if ('settotaluptime' = Arg) then
    begin
      Result := False;
      if Trim(MyGPUID.NodeName) = Trim(StrPas(Stk.PCharStack[Stk.StIdx])) then
           begin
             Result := true;
             MyGPUID.Uptime := 0;
             MyGPUID.TotalUptime := Stk.Stack[Stk.StIdx-1];
             Stk.StIdx := Stk.StIdx-1;
           end;
    end
    else
    if ('iscapable' = Arg) then
    begin
      Result := false;
      if PlugMan.IsCapable(StrPas(Stk.PCharStack[Stk.StIdx])) then
        stk.Stack[stk.StIdx] := 1
      else
        stk.Stack[stk.StIdx] := 0;
      Stk.PCharStack[Stk.StIdx] := nil;
      Result := true;
    end
    else
    if ('isbusy' = Arg) then
    begin
      Result := false;
      if FuncController.isAlreadyCalled(StrPas(Stk.PCharStack[Stk.StIdx])) then
        stk.Stack[stk.StIdx] := 1
      else
        stk.Stack[stk.StIdx] := 0;
      Stk.PCharStack[Stk.StIdx] := nil;
      Result := true;
    end
    else
    if ('whichdll' = Arg) or  ('whichplugin' = Arg) then
    begin
      Result := false;
      PlugMan.WhichPlugin(StrPas(Stk.PCharStack[Stk.StIdx]), tmpSpecial);
      Stk.PCharStack[Stk.StIdx] := nil;
      Stk.StIdx := Stk.StIdx-1;
      Result := LoadStringOnStack(tmpSpecial, Stk);
    end
  end {HandleSpecialCommands};

begin
  JobToRepeat := TJob.Create;
  try
         {gives the actual argument back,
          without any spaces at beginning and at the end.}

    Arg := Trim(ReturnArg(J.StackCommands));
    repeat
      {argument might be a special command}
      if not HandleSpecialCommands(Arg) then
        {argument might be a string}
        if Pos(APOSTROPHE, Arg) = 1 then
        begin
          if Stk.StIdx > (MAXSTACK - 1) then
          begin
            {Job is too big!}
            Stk.StIdx := -1;
            Exit;
          end  {if}
          else
          begin
            Delete(Arg, Length(Arg), 1); {remove ' at the end}
            Delete(Arg, 1, 1); {remove it at the beginning}

            Stk.StIdx := Stk.StIdx + 1;
            Stk.Stack[Stk.StIdx] := INF;
            if Arg <> '' then
            begin
              StrArray[Stk.StIdx] := Arg;
              Stk.PCharStack[Stk.StIdx] := @Arg[1];
            end;
          end; {floating number block}
        end
        else
        {argument might be a float}
        if isFloat(Arg) then
        begin
          if Stk.StIdx > (MAXSTACK - 1) then
          begin
            {Job is too big!}
            Stk.StIdx := -1;
            Exit;
          end  {if}
          else
          begin
            Stk.StIdx := Stk.StIdx + 1;
            Arg := StringReplace (Arg, '.', DecimalSeparator, []);
            Arg := StringReplace (Arg, ',', DecimalSeparator, []);
            Stk.Stack[Stk.StIdx] := StrToFloat(Arg);
            Stk.PCharStack[Stk.StIdx] := nil;
          end; {floating number block}

        end
        else
          {argument might be a plugin call}
        begin
          ResultFunc := False;
                          {a DLL function returns ResultFunc = true
                           only once it finished its job}
          Stk.Progress := 0;
          Stk.Update := False;
          Stk.MultipleResults := False;
          Stk.My := nil;

          // old core
          {core of the plugin system}
          //        repeat
          //          ArgFound := PlugMan.ExecFuncInPlugIns(ResultFunc, Stk, Arg, Plugin);
          //          {send back multiple results}
          //          if Stk.MultipleResults and (Stk.Progress < 100) then
          //            if Assigned(JobForThread.OnFinished) then
          //            begin
          //              JobForThread.Result := StackToStr(JobForThread.Stack);
          //              JobForThread.ComputedTime := Time - JobForThread.ComputedTime;
          //              synchronize (SyncOnJobFinished);
          ////            JobForThread.OnFinished(JobForThread);
          //            end;
          //            {update window panel if necessary}
          //            if UpdateNeeded and
          //            (Stk.Update and (Plugin <> THandle(0)))
          //            then  Synchronize(Update);

          //        until (Stk.Progress = 0) or (Stk.Progress >= 100) or Terminated;


          // new core (rene)
          {core of the plugin system}
          theFunction := PlugMan.FindFunction(Arg, Plugin, PluginName);
          ArgFound    := Assigned(theFunction);
          if ArgFound then
          begin
           // if another thread is already in this function, we prefer to wait
           // in this loop
           while FuncController.isAlreadyCalled(Arg) do Sleep(57);
           // we protect the DLL function call registering its name
           FuncController.registerFunctionCall(arg, pluginName, FuncThreadID);
           // we call the function in the core loop multiple times
           repeat
              //try
               ResultFunc := theFunction^(stk);
              {
                // the idea of this block is to prevent an exception
                // to block the execution of a particular function
                // when it created an exception once

                //TODO: unregistering does not work, if similar jobs
                // are running at the same time
              except
                  on E: Exception do
                    begin
                      FuncController.unregisterFunctionCall(FuncThreadID);
                      raise;
                    end;
              end;
              }
              {send back multiple results}
              if Stk.MultipleResults and (Stk.Progress < 100) then
                if Assigned(JobForThread.OnFinished) then
                begin
                  JobForThread.Result :=
                    StackToStr(JobForThread.Stack);
                  JobForThread.ComputedTime :=
                    Time - JobForThread.ComputedTime;
                  SyncOnJobFinished;
                end;
              {update window panel if necessary}
              if UpdateNeeded and Stk.Update and (Plugin <> THandle(0)) then
              begin
                Update;
              end;

            until (Stk.Progress = 0) or (Stk.Progress >= 100) or Terminated;
            // we deregister the function
            FuncController.unregisterFunctionCall(FuncThreadID);


            //loop QStack see if there are any results
            for i := 1 to MAXSTACK do
              if Assigned(Stk.QCharStack[i]) then
              begin
  {         if Assigned (Stk.PCharStack[i]) then
              ReAllocMem (Stk.PCharStack[i], 1 + length(Stk.QCharStack[i]))
            else
              GetMem (Stk.PCharStack[i], 1 + length(Stk.QCharStack[i]));
             Move (Stk.QCharStack[i]^, Stk.PCharStack[i]^, 1+length(Stk.QCharStack[i]));
  }
                StrArray[i] := Stk.QCharStack[i];
                Stk.PCharStack[i] := @StrArray[i][1];
              end;
            theFunction := PlugMan.FindFunction('cleanstack', Plugin, PluginName);
            //dll may now clean up QStack
            if Assigned(theFunction) then
              theFunction^(Stk);
          end;

          // EOF new core

          {old plugins do not change the variable Stk.Progress}
          {last result is not a multiple result}
          Stk.MultipleResults := False;

          if not ArgFound then
          begin
            Stk.StIdx := -1;
            Exit;
          end;


          if ArgFound and ((not ResultFunc) or (Stk.Stack[1] =
            WRONG_PACKET_ID)) then
          begin
            Stk.StIdx := -1;
            Exit;
          end;
        end;

      Arg := Trim(ReturnArg(J.StackCommands));
          {implemented like this to avoid that we go through the loop with
           Arg ='' this means trouble with FloatToStr and we get a wrong packet}
    until (Arg = '') or Terminated;
  finally
    JobToRepeat.Free;
  end;
end;

function TComputationThread.StackToStr(var Stk: TStack): string;
var
  i:    integer;
  resS, v: string;
  //        resN : Extended;
begin
  if Stk.StIdx = -1 then
  begin
    Result := WRONG_GPU_COMMAND;
    Exit;
  end;

  resS := '';
  for i := 1 to Stk.StIdx do
  begin
    //if (Stk.PCharStack[i]=nil) then
  if (Stk.Stack[i] <> INF) then
    begin
      {$IFDEF D7}
      resS := resS + ',' + FloatToStr(Stk.Stack[i], FormatSet.fs);
      {$ELSE}
      v := FloatToStr(Stk.Stack[i]);
      v := StringReplace (v, DecimalSeparator, '.', []);
      resS:=resS+','+v;
      {$ENDIF}
    end
    else
    // if (Stk.Stack[i]=INF) then
    if Assigned(Stk.PCharStack[i]) then
      resS := resS + ',' + APOSTROPHE + string(Stk.PCharStack[i]) + APOSTROPHE
    else
      resS := resS + ','+ APOSTROPHE+'Plugin error: INF flag set, but Stk.PCharStack['+IntToStr(i)+'] not defined'+APOSTROPHE;
  end;
  Delete(resS, 1, 1);
  Result := resS;
end;

procedure TComputationThread.Update;
begin
  PlugMan.CallUpdateFunction(Plugin, JobForThread.Stack);
end;

procedure TComputationThread.SendResultCallback(stk: PStack);
var
  tmp: boolean;
begin
  //measure to adapt traffic bandwidth
  //before sending back a (multiple) result
  //we sleep a number of milliseconds
  //equal to following formula which depends
  // by download traffic (in bytes/s)
  sleep(Trunc(Math.Max(ct_traffic_download - 2000, 0)));

  //tmp := TStack(stk^).MultipleResults;
  //TStack(stk^).MultipleResults := True;
  tmp := JobForThread.Stack.MultipleResults;
  JobForThread.Stack.MultipleResults := True;
  if Assigned(JobForThread.OnFinished) then
  begin
    JobForThread.Result := StackToStr(stk^{JobForThread.Stack});
    JobForThread.ComputedTime := Time - JobForThread.ComputedTime;
    SyncOnJobFinished;
  end;
  //TStack(stk^).MultipleResults := tmp;
  JobForThread.Stack.MultipleResults := tmp;
end;

procedure SendCallBack(stk: PStack);  stdcall;
begin
  if Assigned(stk) and Assigned(stk^.Thread) and (stk^.Thread is
    TComputationThread) then
    TComputationThread(stk^.Thread).SendResultCallback(stk);
end;

procedure SendStack; stdcall;
var
  i, h: integer;
  ct:   TComputationThread;
begin
  //fetch current thread
  //search appropiate thread
  //if found, send back results
  h  := GetCurrentThreadID;
  ct := nil;
  with TL.LockList do
  begin
    for i := 0 to Count - 1 do
      if TComputationThread(Items[i]).MyThreadID = h then
      begin
        ct := TComputationThread(Items[i]);
        break;
      end;
  end;
  TL.UnlockList;
  if Assigned(ct) then
  begin
    ct.SendResultCallback(@(ct.JobForThread.Stack));
  end;
end;

procedure TComputationThread.CleanStackMem(var stk: TStack);
var
  i: integer;
begin
  for i := 1 to MAXSTACK do
    if stk.PCharStack[i] <> nil then
      {try
        try
          l := length(stk.PCharStack[i]) + 1;
          FreeMem (stk.PCharStack[i], l);
        except end;}
      stk.PCharStack[i] := nil;
  {      except end;}
end;

initialization
  TL := TThreadList.Create;

finalization
  TL.Free;
end.
