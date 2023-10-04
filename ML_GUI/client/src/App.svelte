<script>
  let message = "";
  let src = "";
  async function getName() {
    let thisMessage = message;
    let res = await fetch(`./message?name=${message}`);
    let message_received = await res.text();
    if (res.ok && thisMessage == message) {
      src = message_received;
    }
  }
</script>
<h1>Received Message: {src}!</h1>
<input type="text" placeholder="enter your name" bind:value="{message}" />
<button on:click="{getName}">Send Message</button>
