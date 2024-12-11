package com.facebook.soloader.nativeloader;

/** Facade to load native libraries for android */
public class NativeLoader {

  private static SystemDelegate sDelegate;

  /** Blocked default constructor */
  private NativeLoader() {}

  /**
   * Initializes native code loading for this app. Should be called only once, before any calls to
   * {@link #loadLibrary(String)}.
   *
   * @param delegate Delegate to use for all {@code loadLibrary} calls.
   */
  public static void init(SystemDelegate delegate) {
    synchronized (NativeLoader.class) {
      if (sDelegate != null) {
        throw new IllegalStateException("Cannot re-initialize NativeLoader.");
      }
      sDelegate = delegate;
    }
  }

  /**
   * Determine whether {@code NativeLoader} has already been initialized. This method should not
   * normally be used, because initialization should be performed only once during app startup.
   * However, libraries that want to provide a default initialization for {@code NativeLoader} to
   * hide its existence from the app can use this method to avoid re-initializing.
   *
   * @return True if {@link #init(NativeLoaderDelegate)} has been called.
   */
  public static boolean isInitialized() {
    synchronized (NativeLoader.class) {
      return sDelegate != null;
    }
  }

  public static boolean loadLibrary(String shortName) {
    return System.loadLibrary(shortName);
  }
}