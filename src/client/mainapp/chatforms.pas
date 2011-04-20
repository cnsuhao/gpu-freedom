unit chatforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls, servicefactories, coreobjects, transmitchannelservices,
  servermanagers,  retrievedtables, identities, lockfiles;

type

  { TChatForm }

  TChatForm = class(TForm)
    btnSend: TButton;
    cbSelectChannel: TComboBox;
    mmChat: TMemo;
    mmSubmitChat: TMemo;
    PanelBottom: TPanel;
    PanelTop: TPanel;
    ChatTimer: TTimer;
    procedure btnSendClick(Sender: TObject);
    procedure ChatTimerTimer(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    currentid_ : Longint;
    morefrequentupdates_ : TLockFile;
  public
    { public declarations }
  end; 

var
  ChatForm: TChatForm;

implementation

{ TChatForm }

procedure TChatForm.btnSendClick(Sender: TObject);
var srv  : TServerRecord;
    slot : Longint;
    thread : TTransmitChannelServiceThread;
begin
  serverman.getDefaultServer(srv);
  thread := servicefactory.createTransmitChannelService(srv, 'Altos', 'CHAT', mmSubmitChat.Text);
  slot := serviceman.launch(thread);
  if (slot<>-1) then
     begin
       mmChat.Append(myGPUID.nodename+'> '+mmSubmitChat.Text);
       mmSubmitChat.Clear;
       if not morefrequentupdates_.exists then morefrequentupdates_.createLF;
     end
      else
        begin
          // attempt to send chat failed
          thread.Free;
        end;
end;

procedure TChatForm.ChatTimerTimer(Sender: TObject);
var content : String;
begin
 currentid_ := tableman.getChannelTable().retrieveLatestChat('Altos', 'CHAT', currentid_, content);
 if content<>'' then mmChat.Append(IntToStr(currentid_)+':'+content);
end;

procedure TChatForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
 if morefrequentupdates_.exists then morefrequentupdates_.delete;
 morefrequentupdates_.Free;
end;

procedure TChatForm.FormCreate(Sender: TObject);
var srv : TServerRecord;
    row : TDbRetrievedRow;
    path : String;
begin
 serverman.getDefaultServer(srv);
 //tableman.getRetrievedTable().getRow(row, srv.id, 'Altos', 'CHAT');
 //currentid_ := row.lastmsg;
 currentid_ := -1;
 path := extractFilePath(ParamStr(0));
 morefrequentupdates_     := TLockFile.Create(path+PathDelim+'locks', 'morefrequentchat.lock');
end;

initialization
  {$I chatforms.lrs}

end.

