# -------------------- encoder adapter --------------------
@dataclass
class OneHotResult:
    indices: List[int]
    capitalization_mask: List[str]
    called: str  # which entry point we actually used

async def _maybe_async_call(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

async def zonehot_encode(zh: ZOneHotDB, text: str, *, link_key: Optional[str]) -> OneHotResult:
    """
    Try ZOneHotDB methods in order and normalize outputs.

    Special handling for encode_and_store_text(...) returning None: we then
    read back encoded fields via the public getters.
    """
    tried = []
    last_err: Optional[BaseException] = None

    candidates = ("encode_and_store_text", "encode_text", "encode")
    text_params = ("text_content", "text", "content", "input_text", "document_text")
    id_params = ("document_id", "doc_id", "document_key", "doc_key", "link_key", "id")

    for name in candidates:
        fn = getattr(zh, name, None)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
            params = sig.parameters

            # build kwargs
            kwargs: Dict[str, Any] = {}
            text_param = next((p for p in text_params if p in params), None)
            if text_param:
                kwargs[text_param] = text
            if link_key is not None:
                for p in id_params:
                    if p in params and p not in kwargs:
                        kwargs[p] = link_key
                        break

            # call
            result = await _maybe_async_call(fn, **kwargs) if kwargs else await _maybe_async_call(fn, text)

            # CASE A: encode_and_store_text ⇒ may return None (store-only).
            if name == "encode_and_store_text" and result is None:
                # fetch back encoded fields using public API
                indices = await zh.get_encoded_indices(link_key or "")
                mask = await zh.get_capitalization_mask(link_key or "")
                if indices:
                    log.debug(
                        "zonehot: %s stored ok; fetched indices=%d mask=%d (kwargs=%s)",
                        name, len(indices), len(mask), {k: type(v).__name__ for k, v in kwargs.items()}
                    )
                    return OneHotResult(indices=indices, capitalization_mask=mask, called=name)
                raise RuntimeError("encode_and_store_text stored nothing (empty indices)")

            # CASE B: tuple return
            if isinstance(result, tuple) and len(result) == 2:
                indices, mask = result
                return OneHotResult(indices=list(indices or []), capitalization_mask=list(mask or []), called=name)

            # CASE C: dict return
            if isinstance(result, dict):
                indices = (
                    result.get("indices")
                    or result.get("encoded_indices")
                    or result.get("onehot_indices")
                )
                mask = (
                    result.get("mask")
                    or result.get("capitalization_mask")
                    or result.get("caps_mask")
                )
                if indices is None:
                    raise ValueError(f"{name} returned dict but missing indices")
                if mask is None:
                    mask = []
                return OneHotResult(indices=list(indices or []), capitalization_mask=list(mask or []), called=name)

            # Otherwise unsupported
            raise TypeError(f"{name} returned unsupported type: {type(result).__name__}")

        except BaseException as e:
            tried.append(name)
            last_err = e
            continue

    raise RuntimeError(f"Could not encode text with ZOneHotDB; tried {tried}. Last error: {last_err}")
