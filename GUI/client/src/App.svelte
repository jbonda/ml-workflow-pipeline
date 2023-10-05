<script>
  let message = "";
  let alpha = 0.0001;
  let fit_intercept = false;
  let learning_rate = "constant";
  let src = "";
  async function getHype() {
    let thisMessage = message;
    let res = await fetch(`./message?alpha=${message}`);
    let message_received = await res.text();
    if (res.ok && thisMessage == message) {
      src = message_received;
    }
  }
</script>

<h1>Hyperparameters</h1>
<label for="alpha">Alpha</label>
<input
  type="text"
  placeholder="enter alpha (default = 0.0001)"
  bind:value="{message}"
/>
<br />
<label for="fit_intercept">Fit Intercept Sample</label>
<input
  type="checkbox"
  placeholder="fit_intercept"
  name="fit_intercept"
  value="true"
/>
<br />
<label for="learning_rate">Learning Rate</label>
<input
  type="text"
  placeholder="(e.g. constant)"
  name="learning_rate"
/>
<button on:click="{getHype}">NEXT</button>
