unit FormMain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  uCmdBox, process, ExtCtrls, Menus, ActnList, IniPropStorage, ComCtrls,
  EditBtn, FileCtrl,StdCtrls;

type
  TOctaveState = (psProcessing,psWaiting,psContinue);

  TOctaveAttr = record
    ExePath: String;
    CurrentDir: String;
    State: TOctaveState;
  end;

  { TMainForm }

  TMainForm = class(TForm)
    OctavePrompt: TCmdBox;
    OctaveProcess: TProcess;
    ReadOutputIdleTimer: TIdleTimer;
    OGMainMenu: TMainMenu;
    OGActionList: TActionList;
    OGImageList: TImageList;
    ActFileExit: TAction;
    MenuFile: TMenuItem;
    MenuFileExit: TMenuItem;
    OGIniPropStorage: TIniPropStorage;
    DlgLocateOctave: TOpenDialog;
    OGToolBar: TToolBar;
    ActOctaveRestart: TAction;
    MenuOctave: TMenuItem;
    MenuOctaveRestart: TMenuItem;
    MenuOctaveChDir: TMenuItem;
    ActOctaveChDir: TAction;
    OGSplitter: TSplitter;
    PanelWork: TPanel;
    MenuEdit: TMenuItem;
    MenuEditPrefs: TMenuItem;
    DlgChangeDirectory: TSelectDirectoryDialog;
    TBRestart: TToolButton;
    DECurrentDir: TDirectoryEdit;
    TBClearPrompt: TToolButton;
    ActClearPrompt: TAction;
    TBSaveOutput: TToolButton;
    ActSaveOutput: TAction;
    DlgSaveOutput: TSaveDialog;
    MenuFileSave: TMenuItem;
    Separator1: TMenuItem;
    FLBBrowser: TFileListBox;
    HCFiles: THeaderControl;
    HCHistory: THeaderControl;
    LBHistory: TListBox;
    Splitter1: TSplitter;
    procedure FormCreate(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: boolean);
    procedure FormDestroy(Sender: TObject);
    procedure OctavePromptInput(ACmdBox: TCmdBox; Input: String);
    procedure ReadOutputIdleTimerTimer(Sender: TObject);
    procedure OGIniPropStorageRestoreProperties(Sender: TObject);
    procedure OGIniPropStorageSaveProperties(Sender: TObject);
    procedure ActOctaveRestartExecute(Sender: TObject);
    procedure ActFileExitExecute(Sender: TObject);
    procedure ActClearPromptExecute(Sender: TObject);
    procedure ActOctaveChDirExecute(Sender: TObject);
    procedure ActSaveOutputExecute(Sender: TObject);
    procedure DECurrentDirAcceptDirectory(Sender: TObject; var Value: String);
    procedure DECurrentDirKeyPress(Sender: TObject; var Key: char);
    procedure DECurrentDirEnter(Sender: TObject);
    procedure DECurrentDirExit(Sender: TObject);
    procedure FLBBrowserDblClick(Sender: TObject);
    procedure LBHistoryDblClick(Sender: TObject);
    procedure PanelWorkResize(Sender: TObject);
  private
    { private declarations }
    FOctaveAttr: TOctaveAttr;
    procedure StartProcess;
    procedure StartPrompt; inline;
    function GetCurrentOctaveDir: String;
    procedure ChangeOctaveDir(const NewDir: String);
  public
    { public declarations }
  end;

var
  MainForm: TMainForm;

implementation

uses
  Pipes,StrUtils;

const
  SingleQuote = '''';

{ TMainForm public methods and events }

procedure TMainForm.FormCreate(Sender: TObject);
{$ifdef unix}
var
  IniFilePath: String;
{$endif}
begin
  {$ifdef unix}
    IniFilePath:=AppendPathDelim(ExtractFilePath(Application.ExeName));
    OGIniPropStorage.IniFileName:=IniFilePath+'.'+ExtractFileName(Application.ExeName);
  {$endif}
  // Set default colors for writing
  OctavePrompt.TextColors(clBlue,clWhite);
  // The process is not started here, because we need to read initialization file first
end;

procedure TMainForm.FormCloseQuery(Sender: TObject; var CanClose: boolean);
begin
  if FOctaveAttr.State=psProcessing then begin
    ShowMessage('A process is running'+LineEnding+'Please wait until it finishes first');
    CanClose:=false;
  end;
end;

procedure TMainForm.FormDestroy(Sender: TObject);
begin
  with OctaveProcess do
    if Running then Terminate(0);
end;

procedure TMainForm.OctavePromptInput(ACmdBox: TCmdBox; Input: String);
var
  BytesWritten: LongInt;
begin
  if (Input='quit') or (Input='exit') then Close;
  LBHistory.Items.Add(Input);
  { Cmdbox trims new line, so we need to add it manually before passing input to
    Octave process }
  Input:=Input+#10;
  FOctaveAttr.State:=psProcessing;
  BytesWritten:=OctaveProcess.Input.Write(Input[1],Length(Input));
  if (BytesWritten<>Length(Input)) then ShowMessage('Something''s wrong... not '+
    'all input are written');
end;

procedure TMainForm.ReadOutputIdleTimerTimer(Sender: TObject);

  procedure ProcessStream(Stream: TInputPipeStream);
  var
    Buffer: String;
    BytesRead: Integer;
    BytesAvailable: LongWord;
    n: LongInt;
  begin
    if OctaveProcess.Running then begin
      with Stream do begin
        BytesAvailable:=NumBytesAvailable;
        BytesRead:=0;
        while BytesAvailable>0 do begin
          SetLength(Buffer,BytesRead+BytesAvailable);
          Inc(BytesRead,Read(Buffer[BytesRead+1],BytesAvailable));
          BytesAvailable:=NumBytesAvailable;
        end;
      end;
      with FOctaveAttr do
        if State=psProcessing then begin
          n:=Pos('<?>',Buffer);
          if n<>0 then begin
            State:=psWaiting;
            Delete(Buffer,n,3);
            StartPrompt;
          end else begin
            n:=Pos('<!>',Buffer);
            if n<>0 then begin
              State:=psContinue;
              Delete(Buffer,n,3);
              StartPrompt;
            end;
          end;
        end;
      if Buffer<>'' then
        OctavePrompt.Write(Copy(Buffer,1,BytesRead));
    end;
  end;

begin
  // Separate StdOut and StdErr handler, so each can have its own color set
  ProcessStream(OctaveProcess.Output);
  OctavePrompt.TextColors(clRed,clWhite);
  ProcessStream(OctaveProcess.StdErr);
  OctavePrompt.TextColors(clBlue,clWhite);
end;

procedure TMainForm.OGIniPropStorageRestoreProperties(Sender: TObject);
var
  LastDir: String;
begin
  with FOctaveAttr do begin
    with OGIniPropStorage do begin
      IniSection:='Octave';
      ExePath:=ReadString('ExePath','');
      LastDir:=ReadString('LastDir',ExtractFilePath(Application.ExeName));
      if DirectoryExists(LastDir) then
        CurrentDir:=LastDir
      else
        CurrentDir:=ExtractFilePath(Application.ExeName);
      IniSection:='GUI';
    end;
    { Either initialization file not yet exists or the user set OctaveExePath value
      to empty }
    if (ExePath='') or not FileExists(ExePath) then begin
      MessageDlg('Octave executable not set or invalid','You haven''t told the program where '+
        'to look for GNU Octave executable or it points to a non-existent file'+LineEnding+
        'You will be able to choose after clicking OK',mtInformation,[mbOK],0);
      // This will trigger GetGNUOctaveExe in StartProcess
      ExePath:='';
    end;
    DECurrentDir.Directory:=CurrentDir;
    FLBBrowser.Directory:=CurrentDir;
    State:=psProcessing;
    // Start Octave process after reading initializatio file
    StartProcess;
  end;
end;

procedure TMainForm.OGIniPropStorageSaveProperties(Sender: TObject);
begin
  with OGIniPropStorage do begin
    IniSection:='Octave';
    with FOctaveAttr do begin
      WriteString('ExePath',ExePath);
      WriteString('LastDir',GetCurrentOctaveDir);
    end;
    IniSection:='GUI';
  end;
end;

procedure TMainForm.ActOctaveRestartExecute(Sender: TObject);
begin
  if QuestionDlg('Confirm restarting Octave',PadCenter('Are you sure about it?',37)+
    LineEnding+'You will lose all current information',mtConfirmation,[mrYes,mrNo],
    0)=mrYes then begin
    // Make sure the timer handler doesn't try to read from a dead process
    ReadOutputIdleTimer.Enabled:=false;
    with OctaveProcess do begin
      // Save last active directory
      CurrentDirectory:=GetCurrentOctaveDir;
      if Running then Terminate(0);
      Execute;
    end;
    OctavePrompt.Clear;
    StartPrompt;
    ReadOutputIdleTimer.Enabled:=true;
  end;
end;

procedure TMainForm.ActFileExitExecute(Sender: TObject);
begin
  Close;
end;

procedure TMainForm.ActOctaveChDirExecute(Sender: TObject);
begin
  if DlgChangeDirectory.Execute then
    ChangeOctaveDir(DlgChangeDirectory.FileName);
end;

procedure TMainForm.ActClearPromptExecute(Sender: TObject);
begin
  OctavePrompt.Clear;
end;

procedure TMainForm.ActSaveOutputExecute(Sender: TObject);
begin
  if DlgSaveOutput.Execute then
    OctavePrompt.SaveToFile(DlgSaveOutput.FileName);
end;

procedure TMainForm.DECurrentDirAcceptDirectory(Sender: TObject;
  var Value: String);
begin
  ChangeOctaveDir(Value);
end;

procedure TMainForm.DECurrentDirKeyPress(Sender: TObject; var Key: char);
begin
  if Key=#13 then
    ChangeOctaveDir(DECurrentDir.Directory);
end;

procedure TMainForm.DECurrentDirEnter(Sender: TObject);
begin
  DECurrentDir.Hint:=DECurrentDir.Directory;
end;

procedure TMainForm.DECurrentDirExit(Sender: TObject);
begin
  DECurrentDir.Hint:='Current Octave directory';
end;

procedure TMainForm.FLBBrowserDblClick(Sender: TObject);
var
  EditCmd: String = 'edit ';
begin
  if FOctaveAttr.State<>psProcessing then begin
    FOctaveAttr.State:=psProcessing;
    EditCmd:=EditCmd+SingleQuote+FLBBrowser.FileName+SingleQuote+#10;
    OctaveProcess.Input.Write(EditCmd[1],Length(EditCmd));
  end;
end;

procedure TMainForm.LBHistoryDblClick(Sender: TObject);
var
  Cmd: String;
begin
  Cmd:=LBHistory.Items.Strings[LBHistory.ItemIndex]+#10;
  if FOctaveAttr.State<>psProcessing then begin
    FOctaveAttr.State:=psProcessing;
    OctaveProcess.Input.Write(Cmd[1],Length(Cmd));
  end;
end;

procedure TMainForm.PanelWorkResize(Sender: TObject);
begin
  HCFiles.Sections.Items[0].Width:=PanelWork.Width;
  HCHistory.Sections.Items[0].Width:=PanelWork.Width;
end;

{ TMainForm private methods }

procedure TMainForm.StartProcess;

  function IsGNUOctaveExe: Boolean;
  var
    GNUOctaveStr: String;
  begin
    with OctaveProcess.Output do begin
      // Wait for the first 10 bytes
      while NumBytesAvailable<10 do ;
      SetLength(GNUOctaveStr,10);
      Read(GNUOctaveStr[1],10);
      // It should be 'GNU Octave' (without quotes)
      Result:=GNUOctaveStr='GNU Octave';
    end;
  end;

  function GetGNUOctaveExe: Boolean;
  begin
    if not DlgLocateOctave.Execute then begin
      MessageDlg('You haven''t chosen any GNU Octave executable'+LineEnding+
        'Please set it later under Octave->Choose Octave executable',mtInformation,[mbOK],0);
      Result:=false;
    end else begin
      FOctaveAttr.ExePath:=DlgLocateOctave.FileName;
      Result:=true;
    end;
  end;

const
  CmdLineOpts = '--interactive --no-history --eval "PS1(\"<?>\");PS2(\"<!>\");" --persist';
var
  Ready: Boolean;
begin
  Ready:=false;
  while not Ready do begin
    // Check if the user has entered path to Octave executable
    if (FOctaveAttr.ExePath='') and not GetGNUOctaveExe then Exit;
    // Start the process
    with OctaveProcess do begin
      Options:=[poUsePipes,poNoConsole];
      ShowWindow:=swoNone;
      with FOctaveAttr do begin
        CommandLine:='"'+ExePath+'"';
        CurrentDirectory:=CurrentDir;
      end;
      Execute;
      // Check if it's really a GNU Octave executable
      if IsGNUOctaveExe then begin
        Ready:=true;
        // Restart with correct command line options
        if Running then Terminate(0);
        CommandLine:=CommandLine+' '+CmdLineOpts;
        Execute;
        ReadOutputIdleTimer.Enabled:=true;
        StartPrompt;
      end else begin
        MessageDlg('The executable you choose doesn''t seem to be GNU Octave executable'+
          LineEnding+'Please choose the correct one',mtInformation,[mbOK],0);
        with OctaveProcess do
          if Running then Terminate(0);
        GetGNUOctaveExe;
      end;
    end;
  end;
end;

procedure TMainForm.StartPrompt; inline;
begin
  case FOctaveAttr.State of
    psProcessing: ; // simply do nothing, let the process finishes first
    psWaiting   : OctavePrompt.StartRead(clBlack,clWhite,'>',clBlack,clWhite);
    psContinue  : OctavePrompt.StartRead(clBlack,clWhite,'>>',clBlack,clWhite);
  end;
end;

function TMainForm.GetCurrentOctaveDir: String;
const
  PwdCmd = 'pwd'#10;
var
  PwdResult: String;
  BytesAvailable: LongWord;
  BytesRead: LongInt;
begin
  // Make sure the timer handler doesn't interfere with handler below
  ReadOutputIdleTimer.Enabled:=false;
  Sleep(250);
  // Force reading (avoid junks prepended to pwd output)
  ReadOutputIdleTimerTimer(nil);
  // Send 'pwd' to Octave process
  OctaveProcess.Input.Write(PwdCmd[1],Length(PwdCmd));
  GetCurrentOctaveDir:='';
  Sleep(250); // Give the process a chance
  with OctaveProcess.Output do begin
    BytesAvailable:=NumBytesAvailable;
    BytesRead:=0;
    while BytesAvailable>0 do begin
      SetLength(PwdResult,BytesAvailable);
      BytesRead:=Read(PwdResult[1],BytesAvailable);
      GetCurrentOctaveDir:=GetCurrentOctaveDir+Copy(PwdResult,1,BytesRead);
      BytesAvailable:=NumBytesAvailable;
    end;
  end;
  { The first 6 chars are 'ans = ' which we don't need. Therefore, copy it
    starting from the 7th char }
  GetCurrentOctaveDir:=Copy(GetCurrentOctaveDir,7,Length(GetCurrentOctaveDir)-7);
  // Delete trailing <?>
  Delete(GetCurrentOctaveDir,Length(GetCurrentOctaveDir)-3,4);
end;

procedure TMainForm.ChangeOctaveDir(const NewDir: String); inline;
var
  CdStr: String;
begin
  CdStr:='cd '+SingleQuote+NewDir+SingleQuote+#10;
  OctavePrompt.WriteLn('Changing Octave directory to "'+NewDir+'" ...');
  FOctaveAttr.State:=psProcessing;
  OctaveProcess.Input.Write(CdStr[1],Length(CdStr));
  DECurrentDir.Directory:=NewDir;
  FLBBrowser.Directory:=NewDir;
end;

initialization
  {$I FormMain.lrs}

end.

