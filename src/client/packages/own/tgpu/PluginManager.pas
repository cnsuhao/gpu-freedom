{$DEFINE MSWINDOWS}
{$IFDEF VER170}
{$DEFINE D7}
{$ENDIF}
{$IFDEF VER150}
{$DEFINE D7}
{$ENDIF}
unit PluginManager;
{
 The PluginManager handles a linked list of Plugins, which are DLLs
 containing computational algorithms. DLLs are a collection of
 method calls, each call contains a function with the computational
 algorithm. The function calls all have the same signature defined
 in Definitons class.
 
 PluginManager can load and unload them dinamycally at runtime.
 It can find functions inside the DLL, and execute them.
 It can retrieve a list of loaded plugins and tell if a particular
 plugin is loaded or not. 

 ComputationThread uses the PluginManager to retrieve function
 calls and executes them.
 
}

interface

uses {$IFDEF MSWINDOWS} Windows, FileCtrl,{$ENDIF}
  SysUtils, definitions, common, utils;

type
  {list used to load plugins}
  PPluginList = ^TPluginList;
  {list element}
  TPluginList = record
    DLL:  THandle;
    Name: string;
    Next: PPluginList;
  end;



type
  PPluginManager = ^TPluginManager;

  TPluginManager = class(TObject)
  private
    Plugins,                      {list with all loaded dlls}
    tmpList: PPluginList;   {temporary pointer used by GetPluginsName}

  public
    Directory: string;
    constructor Create;
    procedure BeforeDestruction; override;

    procedure LoadAllPlugins(Extension: string);
    procedure UnloadAllPlugins;

    function LoadSinglePlugin(DLLName: string): boolean;
    function UnloadSinglePlugin(DLLName: string): boolean;
    // p contains the PlugElement, if plugin is found,
    // else it will be nil
    function isPluginAlreadyLoaded(DLLName : String) : boolean; overload;
    function isPluginAlreadyLoaded(DLLName : string;
                                   var p : PPluginList) : boolean; overload;
                                   
    procedure PluginNamesTop;
    function GetPluginsName: string;
    function ExecFuncInPlugIns(var ResFunc: boolean; var Stk: TStack;
      Arg: string; var PluginUsed: THandle): boolean;
    procedure GetPluginDescription(DLLString: string; var Description: PChar;
      var WebLinkToPlugin: string);
    function CallUpdateFunction(DLL: THandle; stk: TStack): boolean;
    function FindFunction(FunctionName: string;
      var PluginUsed: THandle): TDLLFunction;  overload;
    function FindFunction(FunctionName: string;
      var PluginUsed: THandle; var pluginName : String): TDLLFunction; overload;
    
    // returns true if a function is present in one of the dlls
    function isCapable(FunctionName : String) : Boolean;
    
    // which plugin contains a given function
    procedure WhichPlugin(FunctionName: string; var pluginName : String);  

end;



implementation


 {*************************************************************}
 {* Plugin Manager                                            *}
 {*************************************************************}

constructor TPluginManager.Create;
begin
  inherited Create;
  PlugIns := nil;
end;

procedure TPluginManager.LoadAllPlugins(Extension: string);
var
  SRec:   TSearchRec;
  retval: integer;
  HDll:   THandle;
  buf:    array [0..1024] of char;
  PlugElement: PPluginList;
begin
  retval := FindFirst(Directory + PLUG_INS + '*.' + Extension, faAnyFile, SRec);
  while retval = 0 do
  begin
    if (SRec.Attr and (faDirectory or faVolumeID)) = 0 then
          (* we found a file, not a directory or volume label,
             log it. Bail out if the log function returns false. *)
    begin
      {here we try to load a plugin with SRec.Name}
      hDLL := LoadLibrary(StrPCopy(buf, Directory + PLUG_INS + SRec.Name));
      if hDLL <> 0 then
      begin
        {we attach new element at the front of the list}
        New(PlugElement);
        PlugElement.Dll := hDll;
        PlugElement.Next := Plugins;
        PlugElement.Name := SRec.Name;
        Plugins := PlugElement;
      end;
    end;
    retval := FindNext(SRec);
  end;
  //SysUtils.FindClose(SRec);     { added for Win32 compatibility }

  pluginNamesTop;
end;

procedure TPluginManager.UnLoadAllPlugins;
var
  p, tmp: PPluginList;
begin
  //free all plugins
  p := PlugIns;
  while Assigned(p) do
  begin
    PlugIns := p;
    FreeLibrary(p.DLL);
    tmp := p;
    p   := p.Next;
    Dispose(tmp);
  end;
  Plugins := nil;
end;

function TPluginManager.LoadSinglePlugin(DLLName: string): boolean;
var
  PlugElement, p: PPluginList;
  retval: integer;
  SRec:   TSearchRec;
  HDll:   THandle;
  buf:    array [0..1024] of char;

begin
  Result := False;
  if isPluginAlreadyLoaded(DLLName, p) then
    begin
     Result := True;
     Exit;
    end;

  //find the file
  retval := FindFirst(Directory + PLUG_INS + DLLName, faAnyFile, SRec);
  if retval <> 0 then
    Exit;

  if (SRec.Attr and (faDirectory or faVolumeID)) = 0 then
  begin
    //here we try to load a plugin with SRec.Name}
    hDLL := LoadLibrary(StrPCopy(buf, Directory + PLUG_INS + SRec.Name));
    if hDLL <> 0 then
    begin
      //we attach new element at the front of the list}
      New(PlugElement);
      PlugElement.Dll := hDll;
      PlugElement.Next := Plugins;
      PlugElement.Name := SRec.Name;
      Plugins := PlugElement;
      Result  := True;
    end;
  end;
end;

function TPluginManager.UnloadSinglePlugin(DLLName: string): boolean;
var
  found: boolean;
  p, q:  PPluginList;
begin
  Result := False;
  if not isPluginAlreadyLoaded(DLLName, p) then
    begin
     Result := True;
     Exit;
    end;

  //p contains the PluginElement to be unloaded
  //now we go again through the list to find the element
  //before p
  if (p = Plugins) then
  begin
    //p is first element so there is no element before
    //we advance the beginning of the list by one
    Plugins := p.Next;
  end
  else
  begin
    q := Plugins;
    while Assigned(q) do
    begin
      if (q.Next = p) then
        break;
      q := q.Next;
    end;
    //here is q previous element
    //now we exclude p from the chain
    q.Next := p.Next;
  end;

  //finally we dispose p
  FreeLibrary(p.DLL);
  Dispose(p);
  Result := True;
end;


function TPluginManager.isPluginAlreadyLoaded(DLLName : string) : boolean;
var p : PPluginList;
begin
  Result := isPluginAlreadyLoaded(DLLName, p);
end;

function TPluginManager.isPluginAlreadyLoaded(DLLName : string;
                                              var p : PPluginList) : boolean;
begin
  p      := PlugIns;
  Result  := False;
  while Assigned(p) do
  begin
    if (p.Name = DLLName) then
    begin
      Result := True;
      Exit;
    end;
    p := p.Next;
  end;
  p:= nil; // we did not find the plugin
end;


// goes to the top of the list
procedure TPluginManager.PluginNamesTop;
begin
  tmpList := Plugins;
end;

function TPluginManager.GetPluginsName: string;
begin
  Result := '';
  if tmpList <> nil then
  begin
    Result  := tmpList.Name;
    tmpList := tmpList.Next;
  end;
end;

function TPluginManager.ExecFuncInPlugIns(var ResFunc: boolean;
  var Stk: TStack; Arg: string; var PluginUsed: THandle): boolean;
 {the reference var Resfunc is the result of the plugin function and it is
  true if all went right}
var
  theFunction: TDllFunction;
  p:   PPluginList;
  buf: array [0..144] of char;
begin
  Result := False;
  PluginUsed := THandle(0);
  {use dll to find if the argument is in a dll}
  {try to find the function in all other DLL's which are loaded}
  p := PlugIns;
  while Assigned(p) do
  begin
    theFunction := GetProcAddress(p.DLL, StrPCopy(buf, Arg));
    if Assigned(theFunction) then
    begin
      ResFunc := theFunction(Stk);
      Result := True;
      PluginUsed := p.DLL;
      p := nil; {to exit loop, plugin found}
    end {if}
    else
      p := p.Next;
  end; {while}
end; {SearchInPlugIns}

procedure TPluginManager.GetPluginDescription(DLLString: string;
  var Description: PChar; var WebLinkToPlugin: string);
var
  p: PPluginList;
  ResFunc: PChar;
  theFunction:

       function: PChar;
  buf: array [0..64] of char;
begin

  p := PlugIns;
  while Assigned(p) do
  begin
    if p.Name = DllString then
    begin
      {get address of the description function} @
        theFunction := GetProcAddress(p.DLL, StrPCopy(buf, 'description'));
      {get description}
      if @theFunction <> nil then
      begin
        ResFunc := theFunction;
        if ResFunc <> nil then
        begin
          Description := ResFunc;
        end;
      end;

      {get link to site} @
        theFunction := GetProcAddress(p.DLL, StrPCopy(buf, 'weblinktoplugin'));
      if @theFunction <> nil then
      begin

        ResFunc := theFunction;

        if ResFunc <> nil then
        begin
          {here I'll get errors if the buffer is too little}
          WebLinkToPlugin := StrPCopy(buf, ResFunc);
        end;
      end;
      p := nil;  {to exit loop}
    end
    else
      p := p.Next;
  end;

end;

function TPluginManager.CallUpdateFunction(DLL: THandle; stk: TStack): boolean;
var
  ResFunc:     boolean;
  theFunction: TDLLFunction;
begin
  Result  := False;
  ResFunc := False;
  @theFunction := GetProcAddress(DLL, PChar('update'));
  if @theFunction <> nil then
    try
      ResFunc := theFunction(stk);
    except
      ResFunc := False;
    end;
  Result := ResFunc;
end;


procedure TPluginManager.BeforeDestruction;
begin
  UnloadAllPlugins;
end;

function TPluginManager.FindFunction(FunctionName: string;
  var PluginUsed: THandle; var PluginName : String): TDLLFunction;
var
  p: PPluginList;
begin
  p      := PlugIns;
  Result := nil;
  PluginUsed := THandle(0);
  while Assigned(p) do
  begin
    Result := GetProcAddress(p.DLL, PChar(FunctionName));
    if Assigned(Result) then
    begin
      PluginUsed := p.DLL;
      PluginName := p.Name;
      break;
    end
    else
      p := p.Next;
  end; {while}
end;

function TPluginManager.FindFunction(FunctionName: string;
      var PluginUsed: THandle): TDLLFunction;  
var dummy : String;
begin
 FindFunction(FunctionName, pluginUsed, dummy);
end;

procedure TPluginManager.WhichPlugin(FunctionName: string;
      var pluginName : String);  
var pluginUsed : THandle;
begin
 FindFunction(FunctionName, pluginUsed, pluginName);
 if pluginUsed = (THandle(0)) then pluginName:= '';
end;

function TPluginManager.isCapable(FunctionName : String) : boolean;
var PluginUsed : THandle;
begin
   FindFunction(FunctionName, pluginUsed);
   Result := PluginUsed <> THandle(0);
end;


end.
