{$DEFINE MSWINDOWS}
unit FrontendManager;
{
 FrontendManager defines queues.
 The handling of the queue is still done by TGPUTella
 in gpu_main_form.pas (TODO: this is bad!)

}
interface

uses {$IFDEF MSWINDOWS} Windows, FileCtrl,{$ENDIF}
  SysUtils, definitions, common, utils;

const
  SIZE_CIRC_BUFFER = 100;

type
  // ct_Port: frontend is using UDP/IP channels, we store its UDP port (e.g. 32145)
  // ct_FormName: frontend is using Windows Message channel, we store Frontend Name
  //              (e.g. TNetmapperForm)
  // ct_None: dummy if answer is not needed, e.g if GPU core itself registers in the
  //          queue
  TContactType = (ct_None, ct_Port, ct_FormName);

{record to store frontend information}
type
  TFrontendInfo = record
    RegisteredJobID,     // JobID the frontend sent to GPU Core
    Contact: string;     // information on how to contact
                         // frontend, either port or formname
    TypeID:  TContactType;
  end;



type
  PFrontendManager = ^TFrontendManager;

  TFrontendManager = class(TObject)
  private
    FCIdx: integer;
  public
    FrontendsCircular: array[1..SIZE_CIRC_BUFFER] of TFrontendInfo;

    constructor Create;
    procedure RegisterFrontend(_JobID, _Contact: string; _type: TContactType);
    // GetContactInfo does a backward search in the queue and returns
    // true if it found something
    function GetContactInfo(_JobID: string; var _Contact: string;
      var _type: TContactType): boolean;

    // GetMultipleContactInfo does a linear search through the FrontendsCircular
    // queue, QueuePos variable should be set to 1 for the first call.
    // Unlike GetContactInfo, this function can be called multiple times by
    // GPU core.
    // QueuePos variable is passed by reference and it is used to remember where
    // we should start the next linear search.
    function GetMultipleContactInfo(_JobID: string; var _Contact: string;
      var _type: TContactType; var QueuePos : Longint): boolean;

    procedure UnregisterFrontend(_JobID, _Contact: string; _type: TContactType);
    procedure PrintFrontendStatus;
    procedure BeforeDestruction; override;
  end;



implementation


 {*************************************************************}
 {* Frontend Manager                                          *}
 {*************************************************************}

constructor TFrontendManager.Create;
var
  i: integer;
begin
  inherited Create;
  FCIdx := 1;

  for i := 1 to SIZE_CIRC_BUFFER do
    FrontendsCircular[i].TypeID := ct_None;
end;

procedure TFrontendManager.RegisterFrontend(_JobID, _Contact: string;
  _type: TContactType);
var
  Found: boolean;
  i:     integer;
begin
  {we register stuff only once}
  if _type = ct_None then
    Exit; //these are important. GPU core itself uses ct_None
  Found := False;
  for i := 1 to SIZE_CIRC_BUFFER do
  begin
    if (FrontendsCircular[i].RegisteredJobID = _JobID) and
      (FrontendsCircular[i].Contact = _Contact) and
      (FrontendsCircular[i].TypeID = _type) then
    begin
      Found := True;
      break;
    end;
  end;
  if Found then
    Exit;

  Inc(FCIdx);
  if (FCIdx > SIZE_CIRC_BUFFER) then
    FCIdx := 1;

  FrontendsCircular[FCIdx].RegisteredJobID := _JobID;
  FrontendsCircular[FCIdx].Contact := _Contact;
  FrontendsCircular[FCIdx].TypeID  := _type;
end;

procedure TFrontendManager.UnregisterFrontend(_JobID, _Contact: string;
  _type: TContactType);
var
  i: integer;
begin
  if _type = ct_None then
    Exit; //these are important. GPU core itself uses ct_None

  for i := 1 to SIZE_CIRC_BUFFER do
  begin
    if (FrontendsCircular[i].RegisteredJobID = _JobID) and
      (FrontendsCircular[i].Contact = _Contact) and
      (FrontendsCircular[i].TypeID = _type) then
    begin
      FrontendsCircular[i].RegisteredJobID := '';
      FrontendsCircular[i].Contact := '';
      FrontendsCircular[i].TypeID  := ct_None;
    end;
  end;
end;


function TFrontendManager.GetContactInfo(_JobID: string; var _Contact: string;
  var _type: TContactType): boolean;
var
  i, stop: integer;
  Found:   boolean;
begin
  _Contact := '';
  _type    := ct_None;
  Found    := False;

  i    := FCIdx;
  stop := FCIdx + 1;
  if stop > SIZE_CIRC_BUFFER then
    stop := 1;

  {backwards search}
  while (i <> stop) do
  begin
    if FrontendsCircular[i].RegisteredJobID = _JobId then
    begin
      _Contact := FrontendsCircular[i].Contact;
      _type    := FrontendsCircular[i].TypeID;
      Found    := True;
      break;
    end;

    Dec(i);
    if (i = 0) then
      i := SIZE_CIRC_BUFFER;
  end;

  Result := Found;
end;


function TFrontendManager.GetMultipleContactInfo(_JobID: string; var _Contact: string;
      var _type: TContactType; var QueuePos : Longint): boolean;
var Found : Boolean;
    i     : Longint;
begin
Result := False;
if (QueuePos<1) or (QueuePos>SIZE_CIRC_BUFFER) then Exit;
for i := QueuePos to SIZE_CIRC_BUFFER do
     if FrontendsCircular[i].RegisteredJobID = _JobId then
       begin
         _Contact := FrontendsCircular[i].Contact;
         _type    := FrontendsCircular[i].TypeID;
         Result    := True;
         break;
       end;
QueuePos := i+1; // store position where we should be for the next function call
end;



procedure TFrontendManager.PrintFrontendStatus;
var
  i: integer;
begin
  for i := 1 to SIZE_CIRC_BUFFER do
    WriteLog(FrontendsCircular[i].RegisteredJobId + '/' +
      FrontendsCircular[i].Contact + '/');
end;

procedure TFrontendManager.BeforeDestruction;
begin

end;
end.
