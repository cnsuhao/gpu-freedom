unit chatforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls, servicefactories, coreobjects, transmitchannelservices,
  servermanagers,  retrievedtables, coreservices, identities, lockfiles;

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
    firstTime_ : Boolean;
    procedure updateChat(ownChat : Boolean);

  public
    { public declarations }
  end; 

var
  ChatForm: TChatForm;

implementation

{ TChatForm }

procedure TChatForm.btnSendClick(Sender: TObject);
var
    slot : Longint;
    thread : TTransmitChannelServiceThread;
begin
  thread := servicefactory.createTransmitChannelService('Altos', 'CHAT', mmSubmitChat.Text);
  slot := serviceman.launch(TCoreServiceThread(thread), 'TransmitChannelService');
  if (slot<>-1) then
     begin
       mmChat.Append(myGPUID.nodename+'> '+mmSubmitChat.Text);
       mmSubmitChat.Clear;
       if not lf_morefrequentupdates.exists then lf_morefrequentupdates.createLF;
     end
      else
        begin
          // attempt to send chat failed
          thread.Free;
        end;
end;

procedure TChatForm.updateChat(ownChat : Boolean);
var content : String;
begin
 currentid_ := tableman.getChannelTable().retrieveLatestChat('Altos', 'CHAT', currentid_, content, ownChat);
 if content<>'' then mmChat.Append(IntToStr(currentid_)+':'+content);
end;

procedure TChatForm.ChatTimerTimer(Sender: TObject);
begin
  updateChat(firstTime_);
  firstTime_ := false;
end;

procedure TChatForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
 if lf_morefrequentupdates.exists then lf_morefrequentupdates.delete;
end;

procedure TChatForm.FormCreate(Sender: TObject);
var srv : TServerRecord;
    row : TDbRetrievedRow;
begin
 serverman.getDefaultServer(srv);
 currentid_ := -1;
 firstTime_ := true;
end;

initialization
  {$I chatforms.lrs}

end.

