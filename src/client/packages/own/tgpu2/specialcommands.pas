unit specialcommands;

interface

uses identities, gpuconstants;

type TSpecialCommand = class(TObject)
 public
   constructor Create(var plugman : TPluginManager; var meth : TMethodController);
   function isSpecialCommmand(arg : String; var specialType : Longint) : boolean;
   //function execSpecialCommand(arg : String; var error : TGPUError) : boolean
 private
   plugman_ : TPluginManager;
   meth_    : TMethodController;   
end;

constructor Create(var plugman : TPluginManager; var meth : TMethodController);
begin
  plugman_ := plugman;
  meth_    := meth;
end;

implementation

function TSpecialCommand.isSpecialCommmand(arg : String; var specialType : Longint) : boolean;
begin
  //TODO: define this
end;



function TSpecialCommand.execSpecialCommand(arg : String; var stk : TStack; var error : TGPUError) : boolean
begin
  //TODO: go through this mess
    if 'sleep' = Arg then
    begin
      if stk.Idx > 0 then
      begin
        Sleep(Round(Stk.Stack[Stk.StIdx] * 1000));
        Stk.StIdx := Stk.StIdx - 1;
      end;
      Result := True;
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
    end;

end;


end.
